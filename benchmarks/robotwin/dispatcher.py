#!/usr/bin/env python3
"""Central episode-level dispatcher for parallel RoboTwin evaluation.

Why this exists
---------------
The old ``parallel_eval.sh`` / ``dlc_parallel_eval.sh`` scheduled a whole
``task|mode`` (all ``test_num`` episodes) as one indivisible queue item, so at
the tail of a run — when fewer live jobs remain than GPUs — most GPUs sat idle
waiting for the last few whole-task jobs to grind through 100 episodes each.

This dispatcher makes the schedulable unit a **single episode** while still
amortizing the expensive per-process environment boot: one worker process lives
for exactly one ``(task, mode)`` assignment and pulls many episodes for it, and
idle GPUs at the tail are allowed to *join* an already-running job (spawn a
duplicate env) to drain its remaining episodes in parallel.

Two layers live here:

* ``Scheduler`` — a pure, lock-guarded in-memory state machine implementing the
  scheduling policy and the seed/commit accounting. It has no networking and is
  what the unit tests and the ``--self-test`` simulation drive directly.
* ``Dispatcher`` — a thin newline-delimited-JSON TCP server wrapping a
  ``Scheduler``, one persistent connection per worker process.

RoboTwin episode facts this relies on (see script/eval_policy.py):

* A scene is fully determined by its integer ``seed``; whether a seed is
  "valid" (the scripted expert can solve it) depends only on ``(seed, task)``,
  not on the policy under test. So a single monotonic per-job seed allocator
  gives global dedup: every raw seed is tried at most once.
* The cheap expert-check happens *before* the expensive policy rollout, giving a
  natural commit point. The commit handshake below uses it to hit exactly
  ``test_num`` episodes with zero overshoot.

Protocol (one request line -> one response line, JSON objects):

    -> {"type":"hello","node":N,"worker":W,"gpu":G,"port":P}      <- {"ok":true}
    -> {"type":"request_task"}                                    <- {"action":"run","task":T,"mode":M} | {"action":"exit"}
    -> {"type":"request_seed"}                                    <- {"seed":N} | {"drain":true}
    -> {"type":"report_probe","seed":N,"valid":false}            <- {"ok":true}
    -> {"type":"request_commit","seed":N}                        <- {"commit":true} | {"commit":false}
    -> {"type":"report_result","seed":N,"success":B,"step_limit_hit":B}  <- {"ok":true}

A worker's job binding is scoped to its connection: ``request_task`` assigns one
job, and the connection is expected to close after a ``drain`` / ``commit:false``
(the supervisor then launches a fresh worker process that connects anew). If a
connection drops mid-episode, ``Scheduler.release`` returns the in-flight seed so
another worker picks up a replacement — no seed is ever double-counted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import socketserver
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

JobKey = Tuple[str, str]  # (task, mode)


# ---------------------------------------------------------------------------
# Scheduler state
# ---------------------------------------------------------------------------


@dataclass
class Job:
    task: str
    mode: str
    order: int  # stable index for deterministic tie-breaking
    target: int  # desired valid episodes (== test_num)
    next_seed: int  # monotonic per-job seed allocator
    done: int = 0  # valid episodes fully rolled out + reported
    committed: int = 0  # valid episodes approved and currently rolling out
    probing: int = 0  # seeds issued whose expert-check outcome is unknown
    live_envs: int = 0  # worker processes currently bound to this job
    started: bool = False
    suc: int = 0  # successful episodes among done
    step_limit_hits: int = 0
    attempts: int = 0  # seeds ever issued (for the max-attempts safety cap)
    assign_count: int = 0  # times a worker was bound to this job (boot-failure guard)
    exhausted: bool = False  # gave up: too many seed attempts without reaching target

    @property
    def remaining(self) -> int:
        """Valid episodes still needed beyond what is done/committed."""
        return self.target - self.done - self.committed

    @property
    def complete(self) -> bool:
        return self.done >= self.target

    @property
    def terminal(self) -> bool:
        """No more work will be scheduled for this job (done or gave up)."""
        return self.complete or self.exhausted


@dataclass
class WorkerCtx:
    """Per-connection worker handle. One worker == one job assignment."""

    wid: int
    node: int = -1
    worker: int = -1
    gpu: int = -1
    port: int = -1
    key: Optional[JobKey] = None  # job it is bound to (counts toward live_envs)
    probe_seed: Optional[int] = None  # seed issued, expert-check pending
    commit_seed: Optional[int] = None  # seed committed, rollout in progress
    departed: bool = False  # live_envs already released (idempotency guard)
    closed: bool = False  # connection fully torn down (active-count idempotency)
    last_seen: float = 0.0  # last RPC time (for hung-worker reclaim)

    @property
    def tag(self) -> str:
        return f"n{self.node}/w{self.worker}@gpu{self.gpu}:{self.port}#{self.wid}"


class Scheduler:
    """Pure, thread-safe scheduling + seed accounting. No networking.

    All public methods take a ``WorkerCtx`` and mutate job state under a single
    lock. The lock is coarse on purpose: request rates are tiny (one round-trip
    per expensive sim episode), so the bottleneck is never here.
    """

    def __init__(
        self,
        jobs: List[JobKey],
        *,
        test_num: int = 100,
        base_seed: int = 0,
        min_remaining_for_dup: int = 8,
        num_slots: int = 10_000,
        no_dup: bool = False,
        results_path: Optional[str] = None,
        per_job_target: Optional[Dict[JobKey, int]] = None,
        max_attempt_factor: float = 50.0,
        max_boot_failures: int = 3,
        append_results: bool = False,
    ) -> None:
        st_seed = 100_000 * (1 + base_seed)  # mirrors RoboTwin main()
        self._jobs: Dict[JobKey, Job] = {}
        for i, key in enumerate(jobs):
            tgt = (per_job_target or {}).get(key, test_num)
            self._jobs[key] = Job(task=key[0], mode=key[1], order=i, target=tgt, next_seed=st_seed)
        self._theta = max(0, int(min_remaining_for_dup))
        self._num_slots = max(1, int(num_slots))
        self._no_dup = bool(no_dup)
        # H3 guard: cap seeds tried per job so a task whose expert-check almost
        # never passes (misconfig / too hard) can't probe forever. 0/None = off.
        self._max_attempt_factor = float(max_attempt_factor) if max_attempt_factor else 0.0
        # Poison-job guard: a job bound this many times without ever producing a
        # single seed request (worker keeps dying during env boot) is quarantined
        # so RESCUE stops feeding the whole fleet into it. 0 = off.
        self._max_boot_failures = max(0, int(max_boot_failures))
        self._lock = threading.Lock()
        self._next_wid = 0
        self._workers: Dict[int, WorkerCtx] = {}
        # Liveness bookkeeping for the watchdog (H1/H2).
        self._active = 0  # open worker connections
        # None until at least one worker has connected AND then all left. Must NOT
        # start at now(): the dispatcher comes up minutes before the first worker
        # (servers load a multi-GB model first), and a non-None value here would
        # let the idle-grace watchdog false-abort the whole run before any worker
        # ever connects (H4). new_worker() clears it; _close_conn_locked() stamps
        # it only when active drops back to 0.
        self._active_zero_since: Optional[float] = None
        self._last_activity = time.time()  # last RPC of any kind
        self._results_path = results_path
        if results_path and not append_results and os.path.exists(results_path) and os.path.getsize(results_path) > 0:
            raise SystemExit(
                f"[dispatcher] refusing to append to non-empty {results_path} (would double-count on restart). "
                "Use a fresh --results path / run id, or pass --append-results to override."
            )
        self._results_fh = open(results_path, "a" if append_results else "w", encoding="utf-8") if results_path else None

    def _max_attempts_for(self, job: Job) -> Optional[int]:
        if self._max_attempt_factor <= 0:
            return None
        return max(job.target, math.ceil(job.target * self._max_attempt_factor))

    # -- worker lifecycle ---------------------------------------------------

    def new_worker(self, node: int = -1, worker: int = -1, gpu: int = -1, port: int = -1) -> WorkerCtx:
        with self._lock:
            ctx = WorkerCtx(wid=self._next_wid, node=node, worker=worker, gpu=gpu, port=port)
            ctx.last_seen = time.time()
            self._next_wid += 1
            self._workers[ctx.wid] = ctx
            self._active += 1
            self._active_zero_since = None
            self._last_activity = ctx.last_seen
            return ctx

    def _cap(self, job: Job) -> int:
        """Max concurrent envs allowed for a job right now."""
        if self._no_dup:
            return 1
        if self._theta <= 0:
            return self._num_slots
        want = max(1, math.ceil(job.remaining / self._theta))
        return min(want, self._num_slots)

    def assign_task(self, ctx: WorkerCtx) -> dict:
        """Decide which job a free worker should boot (or tell it to exit).

        Priority:
          1. RESCUE: a started job with remaining>0 but live_envs==0 (its worker
             died) — urgent, must not stall. Ignores theta.
          2. SPREAD: an unstarted job with the most remaining episodes.
          3. DUP: join an in-progress job to parallelize its tail, gated by
             theta/cap; pick the one with the longest ETA (remaining/live_envs).
          4. else EXIT.
        """
        with self._lock:
            # Any previous binding is void once a worker asks for a new task.
            self._release_locked(ctx)
            ctx.departed = False
            self._quarantine_poison_locked()  # drop un-bootable jobs before RESCUE feeds them workers

            rescue = [
                j for j in self._jobs.values()
                if j.started and not j.exhausted and j.remaining > 0 and j.live_envs == 0
            ]
            if rescue:
                job = min(rescue, key=lambda j: (-j.remaining, j.order))
                return self._bind_locked(ctx, job)

            unstarted = [j for j in self._jobs.values() if not j.started]
            if unstarted:
                job = min(unstarted, key=lambda j: (-j.remaining, j.order))
                return self._bind_locked(ctx, job)

            # DUP candidates: still need episodes, under cap, and worth it (>=theta).
            best = None
            best_score = None
            for j in self._jobs.values():
                if j.remaining <= 0 or j.exhausted or self._no_dup:
                    continue
                if self._theta > 0 and j.remaining < self._theta:
                    continue
                if j.live_envs >= self._cap(j):
                    continue
                score = j.remaining / j.live_envs if j.live_envs > 0 else float("inf")
                cand = (score, -j.order)
                if best_score is None or cand > best_score:
                    best_score, best = cand, j
            if best is not None:
                return self._bind_locked(ctx, best)

            return {"action": "exit"}

    def _bind_locked(self, ctx: WorkerCtx, job: Job) -> dict:
        job.started = True
        job.live_envs += 1
        job.assign_count += 1
        ctx.key = (job.task, job.mode)
        ctx.departed = False
        return {"action": "run", "task": job.task, "mode": job.mode}

    def _quarantine_poison_locked(self) -> None:
        """Give up on a job that has been bound ``_max_boot_failures`` times but
        never produced a single ``request_seed`` (worker keeps crashing during env
        boot) and currently has no live env. Without this, RESCUE — which sorts by
        ``-remaining`` and ignores theta — hands every freed worker straight back
        to the un-runnable job and slowly drains the whole fleet. ``attempts > 0``
        means at least one worker booted and pulled a seed, so a genuinely-working
        (merely slow/failing) job is never quarantined here; the H3 seed cap
        handles that case instead."""
        cap = self._max_boot_failures
        if cap <= 0:
            return
        for j in self._jobs.values():
            if (not j.exhausted and not j.complete and j.attempts == 0
                    and j.done == 0 and j.live_envs == 0 and j.assign_count >= cap):
                j.exhausted = True

    # -- seed / commit / report --------------------------------------------

    def request_seed(self, ctx: WorkerCtx) -> dict:
        with self._lock:
            self._touch_locked(ctx)
            job = self._job_of(ctx)
            if job is None:
                self._depart_locked(ctx)
                return {"drain": True}
            # A still-outstanding probe seed means the worker re-requested without
            # reporting (protocol slip / crash-in-expert-check): reclaim it so
            # `probing` doesn't leak.
            if ctx.probe_seed is not None:
                job.probing -= 1
                ctx.probe_seed = None
            if job.done + job.committed >= job.target or job.exhausted:
                self._depart_locked(ctx)
                return {"drain": True}
            # H3: bail out of a task that keeps failing the expert check forever.
            cap = self._max_attempts_for(job)
            if cap is not None and job.attempts >= cap and job.done + job.committed < job.target:
                job.exhausted = True
                self._depart_locked(ctx)
                return {"drain": True}
            seed = job.next_seed  # monotonic -> globally unique by construction
            job.next_seed += 1
            job.attempts += 1
            job.probing += 1
            ctx.probe_seed = seed
            return {"seed": seed}

    def report_probe(self, ctx: WorkerCtx, seed: int, valid: bool) -> dict:
        """Expert-check failed (invalid seed). Valid probes go via request_commit."""
        with self._lock:
            self._touch_locked(ctx)
            job = self._job_of(ctx)
            if job is not None and ctx.probe_seed is not None:
                job.probing -= 1
            ctx.probe_seed = None
            return {"ok": True}

    def request_commit(self, ctx: WorkerCtx, seed: int) -> dict:
        """Expert-check passed; decide whether this worker should run the rollout."""
        with self._lock:
            self._touch_locked(ctx)
            job = self._job_of(ctx)
            if job is not None and ctx.probe_seed is not None:
                job.probing -= 1
            probed = ctx.probe_seed
            ctx.probe_seed = None
            if job is None or job.done + job.committed >= job.target:
                self._depart_locked(ctx)
                return {"commit": False}
            job.committed += 1
            # Trust our own issued seed, not the client-supplied argument.
            ctx.commit_seed = probed if probed is not None else seed
            return {"commit": True}

    def report_result(self, ctx: WorkerCtx, seed: int, success: bool, step_limit_hit: bool = False) -> dict:
        with self._lock:
            self._touch_locked(ctx)
            job = self._job_of(ctx)
            if job is not None and ctx.commit_seed is not None:
                job.committed -= 1
                job.done += 1
                if success:
                    job.suc += 1
                if step_limit_hit:
                    job.step_limit_hits += 1
                self._persist_result_locked(ctx, job, ctx.commit_seed, success, step_limit_hit)
            ctx.commit_seed = None
            return {"ok": True}

    def heartbeat(self, ctx: WorkerCtx) -> dict:
        """Keep-alive during a long rollout (no other RPC is sent then). Refreshes
        liveness so a legitimately-slow episode is not mistaken for a hung worker
        by reclaim_stalled / the stall watchdog."""
        with self._lock:
            self._touch_locked(ctx)
        return {"ok": True}

    def release(self, ctx: WorkerCtx) -> None:
        """Connection dropped: return in-flight seed and free the env slot."""
        with self._lock:
            self._release_locked(ctx)
            self._close_conn_locked(ctx)

    # -- internal -----------------------------------------------------------

    def _job_of(self, ctx: WorkerCtx) -> Optional[Job]:
        return self._jobs.get(ctx.key) if ctx.key is not None else None

    def _depart_locked(self, ctx: WorkerCtx) -> None:
        """Worker is leaving its job cleanly (drain / commit:false). At this
        point it holds no probe/commit seed — only the live_envs slot."""
        if ctx.departed:
            return
        job = self._job_of(ctx)
        if job is not None and ctx.key is not None:
            job.live_envs -= 1
        ctx.key = None
        ctx.departed = True

    def _release_locked(self, ctx: WorkerCtx) -> None:
        """Idempotent full teardown for disconnect / reassignment: also returns
        any outstanding probe/commit seed so its episode slot is not lost."""
        job = self._job_of(ctx)
        if job is not None:
            if ctx.probe_seed is not None:
                job.probing -= 1
            if ctx.commit_seed is not None:
                job.committed -= 1
            if ctx.key is not None and not ctx.departed:
                job.live_envs -= 1
        ctx.probe_seed = None
        ctx.commit_seed = None
        ctx.key = None
        ctx.departed = True

    def _close_conn_locked(self, ctx: WorkerCtx) -> None:
        """Idempotently drop a worker connection from the active count."""
        if ctx.closed:
            return
        ctx.closed = True
        self._workers.pop(ctx.wid, None)
        self._active = max(0, self._active - 1)
        if self._active == 0:
            self._active_zero_since = time.time()

    def _touch_locked(self, ctx: WorkerCtx) -> None:
        now = time.time()
        ctx.last_seen = now
        self._last_activity = now

    def reclaim_stalled(self, worker_timeout: float) -> int:
        """H2: a worker bound to a job whose socket is still open but that has
        sent no RPC for ``worker_timeout`` (sim hang / GPU wedge) is presumed
        dead — return any in-flight probe/commit and free its env slot so other
        workers can finish the job. The trigger is being *bound* (``ctx.key`` set)
        and silent, not holding a seed: this also covers a worker that wedges
        during env boot, after ``assign_task`` bound it but before its first
        ``request_seed`` (no probe/commit yet) — otherwise that job's sole env is
        stuck forever with no way for RESCUE/dup to step in. A legit boot is far
        shorter than ``worker_timeout``, and rollouts refresh ``last_seen`` via
        ``heartbeat``. A late report from a revived worker is a no-op (unbound)."""
        if worker_timeout <= 0:
            return 0
        now = time.time()
        reclaimed = 0
        with self._lock:
            for ctx in list(self._workers.values()):
                if ctx.departed or ctx.closed:
                    continue
                if ctx.key is not None and (now - ctx.last_seen) > worker_timeout:
                    self._release_locked(ctx)  # returns in-flight, frees live_envs
                    # Treat a wedged worker as gone for the watchdog too, so if it
                    # was the last one the run aborts promptly (idle_grace) rather
                    # than waiting out stall_timeout on its still-open zombie socket.
                    self._close_conn_locked(ctx)
                    reclaimed += 1
        return reclaimed

    def stall_reason(self, stall_timeout: float, idle_grace: float) -> Optional[str]:
        """Return a human string if the run is wedged (nothing will progress),
        else None. Used by the watchdog to exit with an error instead of the
        old silent infinite self-spin (H1/H2)."""
        now = time.time()
        with self._lock:
            if all(j.terminal for j in self._jobs.values()):
                return None
            unfinished = sum(1 for j in self._jobs.values() if not j.terminal)
            if self._active == 0 and self._active_zero_since is not None:
                idle = now - self._active_zero_since
                if idle > idle_grace:
                    return f"no active workers for {int(idle)}s with {unfinished} unfinished job(s)"
            silent = now - self._last_activity
            if stall_timeout > 0 and silent > stall_timeout:
                return f"no dispatcher activity for {int(silent)}s with {unfinished} unfinished job(s)"
        return None

    def _persist_result_locked(self, ctx: WorkerCtx, job: Job, seed: int, success: bool, slh: bool) -> None:
        if self._results_fh is None:
            return
        rec = {
            "ts": time.time(),
            "task": job.task,
            "mode": job.mode,
            "seed": seed,
            "success": bool(success),
            "step_limit_hit": bool(slh),
            "node": ctx.node,
            "worker": ctx.worker,
        }
        self._results_fh.write(json.dumps(rec) + "\n")
        self._results_fh.flush()

    # -- introspection ------------------------------------------------------

    def is_complete(self) -> bool:
        """True when every job is terminal — reached its target OR gave up
        (exhausted). Exhausted counts as terminal so the run always ends."""
        with self._lock:
            return all(j.terminal for j in self._jobs.values())

    def all_targets_met(self) -> bool:
        with self._lock:
            return all(j.complete for j in self._jobs.values())

    def snapshot(self) -> dict:
        with self._lock:
            jobs = []
            for j in self._jobs.values():
                jobs.append(
                    {
                        "task": j.task,
                        "mode": j.mode,
                        "target": j.target,
                        "done": j.done,
                        "committed": j.committed,
                        "probing": j.probing,
                        "live_envs": j.live_envs,
                        "started": j.started,
                        "suc": j.suc,
                        "step_limit_hits": j.step_limit_hits,
                        "remaining": j.remaining,
                        "attempts": j.attempts,
                        "exhausted": j.exhausted,
                    }
                )
            return {
                "jobs": jobs,
                "complete": all(j.terminal for j in self._jobs.values()),
                "active_workers": self._active,
            }

    def write_summary(self, path: str) -> None:
        with self._lock:
            lines = ["task\tmode\tsuccess\tepisodes\tsuccess_rate\tstep_limit_hits\tstatus"]
            for j in self._jobs.values():
                rate = (j.suc / j.done) if j.done else 0.0
                status = "ok" if j.complete else ("exhausted" if j.exhausted else "incomplete")
                lines.append(
                    f"{j.task}\t{j.mode}\t{j.suc}\t{j.done}\t{rate:.4f}\t{j.step_limit_hits}\t{status}"
                )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def close(self) -> None:
        # Under the lock: _persist_result_locked check-then-writes _results_fh
        # inside the lock, so a handler thread still finishing a report_result
        # during shutdown either flushes its line before we close or sees None
        # after — never an AttributeError on a half-closed file.
        with self._lock:
            if self._results_fh is not None:
                self._results_fh.close()
                self._results_fh = None


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        sched: Scheduler = self.server.scheduler  # type: ignore[attr-defined]
        ctx = sched.new_worker()
        try:
            for raw in self.rfile:
                try:
                    msg = json.loads(raw.decode("utf-8").strip() or "{}")
                except json.JSONDecodeError:
                    self._send({"error": "bad_json"})
                    continue
                resp = self._dispatch(sched, ctx, msg)
                self._send(resp)
        except (ConnectionError, OSError):
            pass
        finally:
            sched.release(ctx)

    def _send(self, obj: dict) -> None:
        self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.wfile.flush()

    @staticmethod
    def _dispatch(sched: Scheduler, ctx: WorkerCtx, msg: dict) -> dict:
        t = msg.get("type")
        if t == "hello":
            ctx.node = int(msg.get("node", -1))
            ctx.worker = int(msg.get("worker", -1))
            ctx.gpu = int(msg.get("gpu", -1))
            ctx.port = int(msg.get("port", -1))
            return {"ok": True}
        if t == "request_task":
            return sched.assign_task(ctx)
        if t == "heartbeat":
            return sched.heartbeat(ctx)
        if t == "request_seed":
            return sched.request_seed(ctx)
        if t == "report_probe":
            return sched.report_probe(ctx, int(msg["seed"]), bool(msg.get("valid", False)))
        if t == "request_commit":
            return sched.request_commit(ctx, int(msg["seed"]))
        if t == "report_result":
            return sched.report_result(
                ctx, int(msg["seed"]), bool(msg.get("success", False)), bool(msg.get("step_limit_hit", False))
            )
        return {"error": f"unknown_type:{t}"}


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Dispatcher:
    """TCP front-end around a Scheduler."""

    def __init__(self, scheduler: Scheduler, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.scheduler = scheduler
        self._server = _ThreadingTCPServer((host, port), _Handler)
        self._server.scheduler = scheduler  # type: ignore[attr-defined]
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        return self._server.server_address  # type: ignore[return-value]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def serve_until_complete(self, poll: float = 0.5) -> None:
        self.start()
        try:
            while not self.scheduler.is_complete():
                time.sleep(poll)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self.scheduler.close()


# ---------------------------------------------------------------------------
# Client — used by episode_worker.py and by TCP integration tests.
# ---------------------------------------------------------------------------


# episode_worker exit codes (also the supervisor's relaunch contract).
EXIT_RELAUNCH = 0  # job drained / commit refused -> supervisor starts a fresh worker
EXIT_NO_MORE_WORK = 3  # dispatcher said exit -> supervisor stops this slot
EXIT_ERROR = 1  # unexpected failure -> supervisor stops this slot (surface it)


class DispatcherClient:
    """Thin synchronous client for the newline-JSON protocol.

    One instance == one worker process == one job assignment lifetime: call
    ``hello`` then ``request_task``; if it returns ``run``, loop
    ``request_seed`` / ``report_probe`` / ``request_commit`` / ``report_result``
    until a ``drain`` or ``commit:false``, then ``close``.
    """

    def __init__(
        self, host: str, port: int, timeout: Optional[float] = None, read_timeout: Optional[float] = None
    ) -> None:
        self._sock = socket.create_connection((host, port), timeout=timeout)
        # `timeout` bounded the *connect* only. Per-RPC reads block by default
        # (read_timeout=None): a briefly-busy dispatcher (GC pause / 64-slot burst)
        # must not be mistaken for a dead one and retire the GPU slot — a truly
        # gone dispatcher still surfaces as EOF -> ConnectionError. Previously the
        # connect timeout doubled as a permanent read deadline, so any slow RPC
        # raised socket.timeout -> EXIT_ERROR -> the slot was retired for good.
        self._sock.settimeout(read_timeout)
        self._fh = self._sock.makefile("rwb")

    def _rpc(self, obj: dict) -> dict:
        self._fh.write((json.dumps(obj) + "\n").encode("utf-8"))
        self._fh.flush()
        line = self._fh.readline()
        if not line:
            raise ConnectionError("dispatcher closed the connection")
        return json.loads(line.decode("utf-8"))

    def hello(self, node: int, worker: int, gpu: int, port: int) -> dict:
        return self._rpc({"type": "hello", "node": node, "worker": worker, "gpu": gpu, "port": port})

    def request_task(self) -> dict:
        return self._rpc({"type": "request_task"})

    def request_seed(self) -> dict:
        return self._rpc({"type": "request_seed"})

    def heartbeat(self) -> dict:
        return self._rpc({"type": "heartbeat"})

    def report_probe(self, seed: int, valid: bool = False) -> dict:
        return self._rpc({"type": "report_probe", "seed": int(seed), "valid": bool(valid)})

    def request_commit(self, seed: int) -> dict:
        return self._rpc({"type": "request_commit", "seed": int(seed)})

    def report_result(self, seed: int, success: bool, step_limit_hit: bool = False) -> dict:
        return self._rpc(
            {
                "type": "report_result",
                "seed": int(seed),
                "success": bool(success),
                "step_limit_hit": bool(step_limit_hit),
            }
        )

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            self._sock.close()


# ---------------------------------------------------------------------------
# Optional live status HTTP endpoint (self-contained real-time view).
# ---------------------------------------------------------------------------


class _StatusHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence per-request logging
        pass

    def do_GET(self) -> None:  # noqa: N802
        sched: Scheduler = self.server.scheduler  # type: ignore[attr-defined]
        if self.path.startswith("/api/state"):
            body = json.dumps(sched.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            body = _render_status_html(sched.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _render_status_html(snap: dict) -> str:
    jobs = snap["jobs"]
    total = sum(j["target"] for j in jobs)
    done = sum(j["done"] for j in jobs)
    live = sum(j["live_envs"] for j in jobs)
    rows = []
    for j in sorted(jobs, key=lambda x: (-x["live_envs"], -x["remaining"])):
        pct = (j["done"] / j["target"] * 100) if j["target"] else 0.0
        rows.append(
            f"<tr><td>{j['task']}</td><td>{j['mode']}</td>"
            f"<td>{j['done']}/{j['target']}</td><td>{pct:.0f}%</td>"
            f"<td>{j['live_envs']}</td><td>{j['committed']}</td>"
            f"<td>{j['remaining']}</td><td>{j['suc']}</td></tr>"
        )
    return (
        "<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=2>"
        "<title>RoboTwin dispatcher</title>"
        "<style>body{font:14px monospace;margin:1.5em}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:2px 8px}</style>"
        f"<h3>episodes {done}/{total} &nbsp; live envs {live} &nbsp; "
        f"{'COMPLETE' if snap['complete'] else 'running'}</h3>"
        "<table><tr><th>task</th><th>mode</th><th>done</th><th>%</th>"
        "<th>live_envs</th><th>committed</th><th>remaining</th><th>suc</th></tr>"
        + "".join(rows)
        + "</table>"
    )


class StatusServer:
    def __init__(self, scheduler: Scheduler, host: str = "0.0.0.0", port: int = 0) -> None:
        self._server = ThreadingHTTPServer((host, port), _StatusHandler)
        self._server.scheduler = scheduler  # type: ignore[attr-defined]

    @property
    def address(self) -> Tuple[str, int]:
        return self._server.server_address  # type: ignore[return-value]

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------------------
# Simulation self-test — models env-boot cost B and episode time E so the
# theta/dup/tail-fill behavior is actually exercised, without RoboTwin.
# ---------------------------------------------------------------------------


def _simulate(
    *,
    num_slots: int,
    tasks: List[str],
    modes: List[str],
    test_num: int,
    boot_cost: float,
    episode_time: float,
    expert_time: float,
    valid_prob: float,
    success_prob: float,
    theta: int,
    no_dup: bool,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Discrete-event simulation of the whole fleet against a real ``Scheduler``.

    A single global virtual clock drives everything: each slot is a generator
    mirroring the worker lifecycle (assign_task -> boot -> loop{request_seed ->
    expert -> commit? -> rollout -> report}), yielding the *duration* until its
    next scheduler interaction. The driver always advances the earliest-free
    slot, so every scheduler decision is evaluated against the fleet state as of
    that moment — exactly the causality of the real system, but deterministic
    and instant. This is what lets us observe whether idle slots at the tail
    correctly pile onto the last task (dup) vs. sit idle (no-dup).

    Costs modeled: ``boot_cost`` (env construction, paid once per assignment),
    ``expert_time`` (per seed, valid or not), ``episode_time`` (policy rollout,
    valid+committed only).
    """
    import heapq
    import random as _random

    keys: List[JobKey] = [(tk, md) for tk in tasks for md in modes]
    sched = Scheduler(
        keys,
        test_num=test_num,
        base_seed=seed,
        min_remaining_for_dup=theta,
        num_slots=num_slots,
        no_dup=no_dup,
    )
    ran_seeds: Dict[JobKey, list] = {k: [] for k in keys}
    boots = [0] * num_slots
    busy_time = [0.0] * num_slots  # productive time (boot+expert+rollout) per slot
    finish_time = [0.0] * num_slots
    rng = _random.Random(seed)

    def slot_gen(slot_idx: int):
        """Yields the duration until this slot's next scheduler interaction."""
        while True:
            ctx = sched.new_worker(node=0, worker=slot_idx, gpu=slot_idx, port=8848 + slot_idx)
            resp = sched.assign_task(ctx)
            if resp.get("action") != "run":
                sched.release(ctx)
                return
            key = (resp["task"], resp["mode"])
            boots[slot_idx] += 1
            busy_time[slot_idx] += boot_cost
            yield boot_cost  # booting the env
            while True:
                r = sched.request_seed(ctx)
                if r.get("drain"):
                    break
                seed_val = r["seed"]
                busy_time[slot_idx] += expert_time
                yield expert_time  # expert check (every seed pays this)
                if rng.random() >= valid_prob:  # expert failed -> seed wasted
                    sched.report_probe(ctx, seed_val, valid=False)
                    continue
                c = sched.request_commit(ctx, seed_val)
                if not c.get("commit"):
                    break  # target already met; discard this valid scene
                busy_time[slot_idx] += episode_time
                yield episode_time  # policy rollout
                sched.report_result(ctx, seed_val, success=(rng.random() < success_prob))
                ran_seeds[key].append(seed_val)
            sched.release(ctx)

    # Event heap keyed by (next_free_time, slot_idx) -> deterministic tie-break.
    heap: List[Tuple[float, int]] = []
    gens = {}
    for i in range(num_slots):
        g = slot_gen(i)
        gens[i] = g
        try:
            dur = next(g)
            heapq.heappush(heap, (dur, i))
        except StopIteration:
            finish_time[i] = 0.0
    while heap:
        t, i = heapq.heappop(heap)
        try:
            dur = gens[i].send(None)
            heapq.heappush(heap, (t + dur, i))
        except StopIteration:
            finish_time[i] = t

    # ---- validation ----
    problems: List[str] = []
    dup_seeds = 0
    for key in keys:
        s = ran_seeds[key]
        dup_seeds += len(s) - len(set(s))
    if dup_seeds:
        problems.append(f"{dup_seeds} duplicated seeds across workers")
    for j in sched.snapshot()["jobs"]:
        n = len(ran_seeds[(j["task"], j["mode"])])
        if j["done"] != n:
            problems.append(f"{j['task']}|{j['mode']}: done={j['done']} but ran {n}")
        if j["done"] != test_num:
            problems.append(f"{j['task']}|{j['mode']}: done={j['done']} != test_num={test_num}")

    makespan = max(finish_time)
    total_busy = sum(busy_time)
    sched.close()
    result = {
        "ok": not problems,
        "problems": problems,
        "makespan": makespan,
        # Utilization = fraction of fleet-time actually spent working (not idle).
        "utilization": (total_busy / (makespan * num_slots)) if makespan else 0.0,
        "total_boots": sum(boots),
        "jobs": len(keys),
        "episodes": sum(len(ran_seeds[k]) for k in keys),
    }
    if verbose:
        print(json.dumps(result, indent=2))
    return result


def _run_self_test() -> int:
    """Representative fleet scenarios. Each runs with dup ON vs OFF so the tail
    speedup is visible; asserts exact counts + zero duplicate seeds in both."""
    scenarios = [
        dict(name="64 slots / 100 jobs (the motivating case)", num_slots=64,
             tasks=[f"task{i:02d}" for i in range(50)], modes=["demo_clean", "demo_randomized"],
             test_num=100, boot_cost=30.0, episode_time=5.0, expert_time=1.0,
             valid_prob=0.8, success_prob=0.5, theta=8),
        dict(name="8 slots / 3 jobs (slots >> jobs, extreme tail)", num_slots=8,
             tasks=["a", "b", "c"], modes=["demo_clean"],
             test_num=50, boot_cost=20.0, episode_time=4.0, expert_time=0.5,
             valid_prob=0.7, success_prob=0.6, theta=6),
        dict(name="64 slots / 1 job (worst case: 63 would idle)", num_slots=64,
             tasks=["only"], modes=["demo_clean"],
             test_num=200, boot_cost=30.0, episode_time=5.0, expert_time=1.0,
             valid_prob=0.8, success_prob=0.5, theta=8),
    ]
    failed = 0
    for sc in scenarios:
        name = sc.pop("name")
        dup = _simulate(no_dup=False, **sc)
        nod = _simulate(no_dup=True, **sc)
        ok = dup["ok"] and nod["ok"]
        if not ok:
            failed += 1
        speedup = (nod["makespan"] / dup["makespan"]) if dup["makespan"] else 0.0
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}\n"
            f"    episodes={dup['episodes']}  jobs={dup['jobs']}\n"
            f"    no-dup : makespan={nod['makespan']:8.0f}s util={nod['utilization']*100:3.0f}% boots={nod['total_boots']}\n"
            f"    dup    : makespan={dup['makespan']:8.0f}s util={dup['utilization']*100:3.0f}% boots={dup['total_boots']}\n"
            f"    speedup: {speedup:.1f}x"
        )
        for p in dup["problems"] + nod["problems"]:
            print(f"    - {p}")
    print("\nALL PASS" if not failed else f"\n{failed} scenario(s) FAILED")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_jobs(tasks: List[str], modes: List[str]) -> List[JobKey]:
    return [(t, m) for t in tasks for m in modes]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RoboTwin episode-level eval dispatcher")
    ap.add_argument("--self-test", action="store_true", help="run in-process fleet simulation and exit")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tasks", nargs="*", default=[], help="task names")
    ap.add_argument("--modes", nargs="*", default=["demo_clean"], help="task configs")
    ap.add_argument("--test-num", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-remaining-for-dup", type=int, default=8, dest="theta")
    ap.add_argument("--num-slots", type=int, default=10_000)
    ap.add_argument("--no-dup", action="store_true")
    ap.add_argument("--results", default=None, help="results.jsonl path")
    ap.add_argument("--summary", default=None, help="summary.tsv path (written on completion)")
    ap.add_argument("--addr-file", default=None, help="write host:port here for peers to discover")
    ap.add_argument(
        "--advertise-host",
        default=None,
        help="host written to --addr-file for clients (default: bind host, or 127.0.0.1 when binding 0.0.0.0)",
    )
    ap.add_argument("--http-port", type=int, default=0, help="serve live status HTML/JSON (0=off)")
    ap.add_argument("--state-file", default=None, help="periodically write snapshot JSON here")
    ap.add_argument("--done-file", default=None,
                    help="touch this on exit (complete/stall) so all nodes can tear down local workers")
    # Liveness / watchdog (H1/H2/H3): keep the dispatcher from ever hanging silently.
    ap.add_argument("--stall-timeout", type=float, default=1800.0,
                    help="exit (incomplete) if no RPC of any kind for this many seconds (0=off)")
    ap.add_argument("--worker-timeout", type=float, default=1200.0,
                    help="reclaim a worker's in-flight seed if it is silent this long — must exceed one rollout (0=off)")
    ap.add_argument("--idle-grace", type=float, default=120.0,
                    help="exit (incomplete) if zero workers are connected for this long while jobs remain")
    ap.add_argument("--max-attempt-factor", type=float, default=50.0,
                    help="give up on a task after target*factor seed attempts (0=unlimited)")
    ap.add_argument("--append-results", action="store_true",
                    help="append to an existing results.jsonl instead of refusing (risks double-count)")
    args = ap.parse_args(argv)

    if args.self_test:
        return _run_self_test()

    if not args.tasks:
        ap.error("--tasks is required unless --self-test")

    sched = Scheduler(
        _resolve_jobs(args.tasks, args.modes),
        test_num=args.test_num,
        base_seed=args.seed,
        min_remaining_for_dup=args.theta,
        num_slots=args.num_slots,
        no_dup=args.no_dup,
        results_path=args.results,
        max_attempt_factor=args.max_attempt_factor,
        append_results=args.append_results,
    )
    disp = Dispatcher(sched, host=args.host, port=args.port)
    disp.start()
    host, port = disp.address
    print(f"[dispatcher] listening on {host}:{port} jobs={len(args.tasks) * len(args.modes)}")
    if args.addr_file:
        adv = args.advertise_host or ("127.0.0.1" if host in ("0.0.0.0", "::") else host)
        with open(args.addr_file, "w", encoding="utf-8") as f:
            f.write(f"{adv}:{port}\n")

    status = None
    if args.http_port:
        status = StatusServer(sched, host="0.0.0.0", port=args.http_port)
        status.start()
        print(f"[dispatcher] status http on 0.0.0.0:{status.address[1]}")

    def _write_state() -> None:
        if args.state_file:
            tmp = args.state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(sched.snapshot(), f)
            os.replace(tmp, args.state_file)

    # A bare SIGTERM (DLC preemption, `kill`, scheduler stop) would otherwise skip
    # the finally below — no done-file, no summary, no clean server shutdown — and
    # cross-node teardown hinges on that done-file. Route TERM through the same
    # KeyboardInterrupt path Ctrl+C already takes so teardown always runs.
    def _raise_kbint(*_a):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _raise_kbint)
    except ValueError:
        pass  # not on the main thread (embedded use) — SIGINT path still applies

    stalled: Optional[str] = None
    interrupted = False
    try:
        while not sched.is_complete():
            sched.reclaim_stalled(args.worker_timeout)  # H2: free hung workers' seeds
            stalled = sched.stall_reason(args.stall_timeout, args.idle_grace)  # H1/H2 watchdog
            if stalled:
                print(f"[dispatcher] STALL: {stalled} — aborting instead of hanging", flush=True)
                break
            _write_state()
            time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
        print("[dispatcher] interrupted (SIGINT/SIGTERM) — tearing down", flush=True)
    finally:
        _write_state()
        if args.summary:
            sched.write_summary(args.summary)
        if status is not None:
            status.shutdown()
        disp.shutdown()
        if args.done_file:  # signal every node to tear down local workers
            try:
                with open(args.done_file, "w", encoding="utf-8") as f:
                    f.write("stall\n" if stalled else ("interrupted\n" if interrupted else "complete\n"))
            except OSError:
                pass

    if interrupted:
        print("[dispatcher] torn down after interrupt", flush=True)
        return 130
    if stalled:
        print(f"[dispatcher] ABORTED (stall): {stalled}", flush=True)
        return 2
    if not sched.all_targets_met():
        print("[dispatcher] complete with exhausted job(s) — see summary status column", flush=True)
        return 3
    print("[dispatcher] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
