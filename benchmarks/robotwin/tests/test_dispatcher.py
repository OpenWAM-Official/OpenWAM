"""Unit tests for the episode-level RoboTwin eval dispatcher.

These drive the pure ``Scheduler`` state machine directly (no networking, no
RoboTwin) plus one TCP round-trip smoke test. Run with::

    pytest benchmarks/robotwin/tests/test_dispatcher.py
"""

from __future__ import annotations

import json
import os
import socket
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dispatcher as D  # noqa: E402

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _drain_one_worker(sched, key, *, valid, success, ctx=None):
    """Run a fake worker for one job assignment until it drains, using a fixed
    ``valid``/``success`` script (lists popped per episode, or callables)."""
    if ctx is None:
        ctx = sched.new_worker()
        resp = sched.assign_task(ctx)
        assert resp["action"] == "run"
        assert (resp["task"], resp["mode"]) == key
    ran = []
    while True:
        r = sched.request_seed(ctx)
        if r.get("drain"):
            break
        seed = r["seed"]
        is_valid = valid(seed) if callable(valid) else valid
        if not is_valid:
            sched.report_probe(ctx, seed, valid=False)
            continue
        c = sched.request_commit(ctx, seed)
        if not c["commit"]:
            break
        sched.report_result(ctx, seed, success=(success(seed) if callable(success) else success))
        ran.append(seed)
    sched.release(ctx)
    return ran


# ---------------------------------------------------------------------------
# exact-count / dedup
# ---------------------------------------------------------------------------


def test_single_worker_hits_exact_target_all_valid():
    sched = D.Scheduler([("t", "m")], test_num=10)
    ran = _drain_one_worker(sched, ("t", "m"), valid=True, success=True)
    assert len(ran) == 10
    snap = sched.snapshot()["jobs"][0]
    assert snap["done"] == 10 and snap["suc"] == 10
    assert sched.is_complete()


def test_seeds_start_at_st_seed_and_are_contiguous_when_all_valid():
    sched = D.Scheduler([("t", "m")], test_num=5, base_seed=0)
    ran = _drain_one_worker(sched, ("t", "m"), valid=True, success=False)
    assert ran == [100000, 100001, 100002, 100003, 100004]


def test_base_seed_offsets_st_seed():
    sched = D.Scheduler([("t", "m")], test_num=3, base_seed=2)
    ran = _drain_one_worker(sched, ("t", "m"), valid=True, success=False)
    assert ran[0] == 100000 * (1 + 2)


def test_rejection_sampling_skips_invalid_seeds_but_counts_target_valids():
    # Only odd seeds are valid; still must collect exactly test_num valids.
    sched = D.Scheduler([("t", "m")], test_num=4)
    ran = _drain_one_worker(sched, ("t", "m"), valid=lambda s: s % 2 == 1, success=True)
    assert len(ran) == 4
    assert all(s % 2 == 1 for s in ran)
    assert sched.snapshot()["jobs"][0]["done"] == 4


def test_two_workers_same_job_no_duplicate_seeds_and_exact_total():
    sched = D.Scheduler([("t", "m")], test_num=20, min_remaining_for_dup=1, num_slots=2)
    # First worker starts the job.
    c1 = sched.new_worker()
    assert sched.assign_task(c1)["action"] == "run"
    # Second worker: no unstarted jobs left -> should dup onto the same job.
    c2 = sched.new_worker()
    r2 = sched.assign_task(c2)
    assert r2 == {"action": "run", "task": "t", "mode": "m"}

    ran1 = _drain_one_worker(sched, ("t", "m"), valid=True, success=True, ctx=c1)
    ran2 = _drain_one_worker(sched, ("t", "m"), valid=True, success=True, ctx=c2)
    allseeds = ran1 + ran2
    assert len(allseeds) == len(set(allseeds)), "seeds duplicated across workers"
    assert len(allseeds) == 20
    assert sched.snapshot()["jobs"][0]["done"] == 20


# ---------------------------------------------------------------------------
# assign_task priority: spread before dup
# ---------------------------------------------------------------------------


def test_spread_before_dup():
    sched = D.Scheduler([("a", "m"), ("b", "m")], test_num=100, min_remaining_for_dup=1)
    c1 = sched.new_worker()
    r1 = sched.assign_task(c1)
    c2 = sched.new_worker()
    r2 = sched.assign_task(c2)
    # Two distinct unstarted jobs must be handed out before any duplication.
    assert {(r1["task"]), (r2["task"])} == {"a", "b"}
    # Third worker: both started -> now allowed to dup.
    c3 = sched.new_worker()
    r3 = sched.assign_task(c3)
    assert r3["action"] == "run"
    assert r3["task"] in {"a", "b"}


def test_exit_when_nothing_left():
    sched = D.Scheduler([("t", "m")], test_num=3)
    _drain_one_worker(sched, ("t", "m"), valid=True, success=True)
    c = sched.new_worker()
    assert sched.assign_task(c) == {"action": "exit"}


# ---------------------------------------------------------------------------
# theta / cap gating
# ---------------------------------------------------------------------------


def test_no_dup_below_theta():
    # One job, one worker already on it, remaining < theta -> no second env.
    sched = D.Scheduler([("t", "m")], test_num=100, min_remaining_for_dup=8, num_slots=4)
    c1 = sched.new_worker()
    sched.assign_task(c1)
    # Simulate progress so remaining drops below theta.
    for _ in range(95):
        r = sched.request_seed(c1)
        sched.request_commit(c1, r["seed"])
        sched.report_result(c1, r["seed"], success=True)
    # remaining now 5 < theta=8; a free worker must NOT dup (returns exit,
    # because the sole live env will finish it within a boot).
    c2 = sched.new_worker()
    assert sched.assign_task(c2) == {"action": "exit"}


def test_dup_allowed_at_or_above_theta():
    sched = D.Scheduler([("t", "m")], test_num=100, min_remaining_for_dup=8, num_slots=4)
    c1 = sched.new_worker()
    sched.assign_task(c1)
    for _ in range(90):  # remaining -> 10 >= theta
        r = sched.request_seed(c1)
        sched.request_commit(c1, r["seed"])
        sched.report_result(c1, r["seed"], success=True)
    c2 = sched.new_worker()
    assert sched.assign_task(c2)["action"] == "run"


def test_no_dup_mode_never_duplicates():
    sched = D.Scheduler([("t", "m")], test_num=100, no_dup=True, num_slots=4)
    c1 = sched.new_worker()
    sched.assign_task(c1)
    c2 = sched.new_worker()
    # Only one env allowed in strict mode -> second worker exits.
    assert sched.assign_task(c2) == {"action": "exit"}


def test_cap_limits_concurrent_envs():
    # remaining=20, theta=8 -> cap = ceil(20/8) = 3 envs max.
    sched = D.Scheduler([("t", "m")], test_num=20, min_remaining_for_dup=8, num_slots=16)
    ctxs = []
    runs = 0
    for _ in range(16):
        c = sched.new_worker()
        r = sched.assign_task(c)
        if r["action"] == "run":
            runs += 1
            ctxs.append(c)
        else:
            break
    assert runs == 3, f"expected cap=3 concurrent envs, got {runs}"


# ---------------------------------------------------------------------------
# rescue + disconnect recovery
# ---------------------------------------------------------------------------


def test_disconnect_returns_committed_seed_and_frees_env():
    sched = D.Scheduler([("t", "m")], test_num=5)
    c1 = sched.new_worker()
    sched.assign_task(c1)
    r = sched.request_seed(c1)
    sched.request_commit(c1, r["seed"])  # committed +=1
    snap = sched.snapshot()["jobs"][0]
    assert snap["committed"] == 1 and snap["live_envs"] == 1
    sched.release(c1)  # crash mid-rollout
    snap = sched.snapshot()["jobs"][0]
    assert snap["committed"] == 0 and snap["live_envs"] == 0 and snap["done"] == 0


def test_rescue_dead_job_gets_reassigned_and_completes():
    sched = D.Scheduler([("t", "m")], test_num=5)
    c1 = sched.new_worker()
    sched.assign_task(c1)
    r = sched.request_seed(c1)
    sched.request_commit(c1, r["seed"])
    sched.report_result(c1, r["seed"], success=True)  # done=1
    sched.release(c1)  # worker dies with remaining>0, live_envs->0
    # A fresh worker must rescue this started-but-abandoned job (not exit).
    c2 = sched.new_worker()
    assert sched.assign_task(c2) == {"action": "run", "task": "t", "mode": "m"}
    ran = _drain_one_worker(sched, ("t", "m"), valid=True, success=True, ctx=c2)
    assert sched.snapshot()["jobs"][0]["done"] == 5
    assert len(ran) == 4  # 1 already done before the crash


def test_commit_false_at_boundary_discards_extra_valid_scene():
    # target=1: first worker commits+runs it; a second worker that expert-passes
    # a seed must get commit:false (no overshoot).
    sched = D.Scheduler([("t", "m")], test_num=1, min_remaining_for_dup=1, num_slots=2)
    c1 = sched.new_worker()
    sched.assign_task(c1)
    c2 = sched.new_worker()
    sched.assign_task(c2)
    # c1 grabs + commits + finishes the only episode.
    r1 = sched.request_seed(c1)
    assert sched.request_commit(c1, r1["seed"])["commit"] is True
    sched.report_result(c1, r1["seed"], success=True)
    # c2 grabs a seed, expert passes, but target already met -> commit:false.
    r2 = sched.request_seed(c2)
    assert r2.get("drain") is True or "seed" in r2
    if "seed" in r2:
        assert sched.request_commit(c2, r2["seed"])["commit"] is False
    assert sched.snapshot()["jobs"][0]["done"] == 1
    assert sched.is_complete()


# ---------------------------------------------------------------------------
# TCP smoke
# ---------------------------------------------------------------------------


def test_tcp_round_trip():
    sched = D.Scheduler([("t", "m")], test_num=2)
    disp = D.Dispatcher(sched, host="127.0.0.1", port=0)
    disp.start()
    host, port = disp.address
    try:
        s = socket.create_connection((host, port), timeout=5)
        f = s.makefile("rwb")

        def rpc(obj):
            f.write((json.dumps(obj) + "\n").encode())
            f.flush()
            return json.loads(f.readline().decode())

        assert rpc({"type": "hello", "node": 0, "worker": 0, "gpu": 0, "port": 8848})["ok"]
        assert rpc({"type": "request_task"})["action"] == "run"
        ran = 0
        while True:
            r = rpc({"type": "request_seed"})
            if r.get("drain"):
                break
            seed = r["seed"]
            assert rpc({"type": "request_commit", "seed": seed})["commit"] is True
            rpc({"type": "report_result", "seed": seed, "success": True})
            ran += 1
        assert ran == 2
        f.close()
        s.close()
    finally:
        disp.shutdown()
    assert sched.snapshot()["jobs"][0]["done"] == 2


def test_multi_worker_supervisor_contract_over_tcp():
    """Exercise the real fleet contract via DispatcherClient over TCP: several
    slots, each a supervisor loop that opens a fresh connection per worker
    (one job lifetime), relaunches on drain, and stops on action==exit. Mirrors
    exactly what parallel_eval.sh's supervisor + episode_worker do."""
    import threading

    tasks = [(f"t{i}", "demo_clean") for i in range(4)]
    sched = D.Scheduler(tasks, test_num=15, min_remaining_for_dup=3, num_slots=6)
    disp = D.Dispatcher(sched, host="127.0.0.1", port=0)
    disp.start()
    host, port = disp.address
    ran_lock = threading.Lock()
    all_ran = []

    def supervisor(slot):
        while True:  # relaunch loop
            cli = D.DispatcherClient(host, port, timeout=5)
            cli.hello(node=0, worker=slot, gpu=slot, port=8848 + slot)
            task_resp = cli.request_task()
            if task_resp.get("action") != "run":
                cli.close()
                return  # EXIT_NO_MORE_WORK
            while True:  # one job lifetime
                r = cli.request_seed()
                if r.get("drain"):
                    break
                seed = r["seed"]
                # deterministic validity/ success by seed
                if seed % 5 == 0:
                    cli.report_probe(seed, valid=False)
                    continue
                if not cli.request_commit(seed).get("commit"):
                    break
                cli.report_result(seed, success=(seed % 2 == 0))
                with ran_lock:
                    all_ran.append((task_resp["task"], seed))
            cli.close()  # -> supervisor relaunches (claims next task or exits)

    threads = [threading.Thread(target=supervisor, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    try:
        assert sched.is_complete()
        # exactly test_num per job, and no seed run twice across all workers.
        for j in sched.snapshot()["jobs"]:
            assert j["done"] == 15, j
        # Seeds are per-job namespaces (each job starts at st_seed), so dedup on
        # (task, seed): the same integer under different tasks is not a clash.
        assert len(all_ran) == len(set(all_ran)), "duplicate (task,seed) across workers over TCP"
        assert len(all_ran) == 4 * 15
    finally:
        disp.shutdown()


# ---------------------------------------------------------------------------
# full simulation
# ---------------------------------------------------------------------------


def test_simulation_exact_and_no_dup_seeds():
    res = D._simulate(
        num_slots=16,
        tasks=[f"t{i}" for i in range(6)],
        modes=["demo_clean", "demo_randomized"],
        test_num=25,
        boot_cost=10.0,
        episode_time=2.0,
        expert_time=0.5,
        valid_prob=0.75,
        success_prob=0.5,
        theta=5,
        no_dup=False,
        seed=0,
    )
    assert res["ok"], res["problems"]
    assert res["episodes"] == 6 * 2 * 25


def test_simulation_strict_mode():
    res = D._simulate(
        num_slots=8,
        tasks=["a", "b", "c"],
        modes=["demo_clean"],
        test_num=20,
        boot_cost=10.0,
        episode_time=2.0,
        expert_time=0.5,
        valid_prob=0.8,
        success_prob=0.5,
        theta=5,
        no_dup=True,
        seed=1,
    )
    assert res["ok"], res["problems"]
    assert res["episodes"] == 3 * 20
