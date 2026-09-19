"""Manifest, RNG-domain, and progress invariants."""

from __future__ import annotations

import builtins
import json
import random
import sys
import types

import numpy as np
import pytest

from benchmarks.robotwin.eval_manifest import choose_instructions
from benchmarks.utils.eval_manifest import load_manifest, seal_manifest, select_entries, verify_manifest, write_manifest
from benchmarks.utils.rng_domain import GlobalRngDomain, decode_numpy_state, encode_numpy_state
from benchmarks.utils.task_progress import ProgressFileReporter, write_progress


def sample_manifest(count: int = 3) -> dict:
    return seal_manifest(
        {
            "benchmark": "test",
            "task": "task-0",
            "entries": [{"episode": index, "seed": 100 + index} for index in range(count)],
        }
    )


def test_manifest_hash_detects_mutation_and_selects_range(tmp_path):
    manifest = sample_manifest()
    path = write_manifest(tmp_path / "manifest.json", manifest)
    assert load_manifest(path, expected_manifest_hash=manifest["manifest_hash"])["entries"][1]["seed"] == 101
    assert select_entries(manifest, 1, 2) == [
        {"episode": 1, "seed": 101},
        {"episode": 2, "seed": 102},
    ]
    mutated = json.loads(json.dumps(manifest))
    mutated["entries"][1]["seed"] = 999
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(mutated)
    with pytest.raises(ValueError, match="exceeds"):
        select_entries(manifest, 2, 2)


def test_libero_manifest_rebuild_runs_all_resets(tmp_path, monkeypatch):
    from benchmarks.libero import eval_manifest as libero_manifest

    libero_path = tmp_path / "LIBERO"
    init_root = libero_path / "libero/libero/init_files"
    bddl_root = libero_path / "libero/libero/bddl_files"
    (init_root / "problem").mkdir(parents=True)
    (bddl_root / "problem").mkdir(parents=True)
    (init_root / "problem/init.pruned_init").write_bytes(b"init")
    (bddl_root / "problem/task.bddl").write_bytes(b"bddl")
    config = tmp_path / "policy.yml"
    config.write_text("seed: 42\nrng_mode: environment\nreseed_each_trial: false\n")

    task = types.SimpleNamespace(
        name="fake task",
        problem_folder="problem",
        init_states_file="init.pruned_init",
        bddl_file="task.bddl",
    )

    class Suite:
        def get_task(self, task_id):
            assert task_id == 0
            return task

        def get_task_init_states(self, task_id):
            assert task_id == 0
            return [object(), object()]

    libero_module = types.ModuleType("libero.libero")
    libero_module.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"suite": Suite})
    libero_module.get_libero_path = lambda kind: {"init_states": init_root, "bddl_files": bddl_root}[kind]
    package = types.ModuleType("libero")
    package.libero = libero_module
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)

    reset_counts = []

    class Env:
        def __init__(self):
            self.resets = 0
            reset_counts.append(self)

        def seed(self, seed):
            np.random.seed(seed)

        def reset(self):
            self.resets += 1
            np.random.random(3)

        def close(self):
            pass

    single_eval = types.ModuleType("single_eval")
    single_eval._make_env_with_randomization_retries = lambda task, cfg: Env()
    monkeypatch.setitem(sys.modules, "single_eval", single_eval)

    first = libero_manifest.build_manifest(config_path=config, suite="suite", task_id=0, total_trials=4)
    second = libero_manifest.build_manifest(config_path=config, suite="suite", task_id=0, total_trials=4)
    assert second == first
    assert [env.resets for env in reset_counts] == [4, 4]
    assert [entry["trial"] for entry in first["entries"]] == list(range(4))


def test_rng_domain_isolates_policy_and_benchmark_streams():
    domain = GlobalRngDomain.from_seed(29)
    with domain.activate():
        benchmark_first = (random.random(), np.random.random())
    random.seed(71)
    np.random.seed(71)
    policy_first = (random.random(), np.random.random())
    with domain.activate():
        benchmark_second = (random.random(), np.random.random())
    policy_second = (random.random(), np.random.random())

    policy_py, policy_np = random.Random(71), np.random.RandomState(71)
    assert policy_first == (policy_py.random(), policy_np.random())
    assert policy_second == (policy_py.random(), policy_np.random())
    benchmark_py, benchmark_np = random.Random(29), np.random.RandomState(29)
    assert benchmark_first == (benchmark_py.random(), benchmark_np.random())
    assert benchmark_second == (benchmark_py.random(), benchmark_np.random())


def test_instruction_selection_uses_candidate_seed_and_restores_caller_rng():
    calls = []

    def generator(task, episode_infos, max_descriptions):
        calls.append(max_descriptions)
        values = [f"instruction-{index}" for index in range(max_descriptions)]
        random.shuffle(values)
        return [{"seen": values, "unseen": list(reversed(values))}]

    random.seed(7)
    np.random.seed(7)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    first = choose_instructions(
        generator, task="adjust_bottle", candidate_seed=100002, episode_info={"object": "bottle"}
    )
    second = choose_instructions(
        generator, task="adjust_bottle", candidate_seed=100002, episode_info={"object": "bottle"}
    )
    assert first == second
    assert set(first) == {"seen", "unseen"}
    assert calls == [100, 100]
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert all(
        np.array_equal(expected, actual) if isinstance(expected, np.ndarray) else expected == actual
        for expected, actual in zip(numpy_state, current_numpy_state)
    )


def test_manifest_entries_replay_identically_in_reverse_mixed_batches_and_retry():
    reference = np.random.RandomState(42)
    entries, expected = [], {}
    for trial in range(6):
        entries.append({"trial": trial, "pre_reset_numpy_state": encode_numpy_state(reference.get_state())})
        expected[trial] = reference.random(4).tolist()
    manifest = seal_manifest({"benchmark": "libero", "entries": entries})

    replayed = {}
    for start, count in ((4, 2), (1, 3), (0, 1), (4, 2)):
        for entry in reversed(select_entries(manifest, start, count)):
            random.seed(900 + entry["trial"])
            np.random.seed(900 + entry["trial"])
            domain = GlobalRngDomain.from_numpy_state(decode_numpy_state(entry["pre_reset_numpy_state"]))
            with domain.activate():
                value = np.random.random(4).tolist()
            previous = replayed.setdefault(entry["trial"], value)
            assert previous == value == expected[entry["trial"]]
    assert set(replayed) == set(range(6))


def test_progress_file_reporter_forwards_each_changed_snapshot_once(tmp_path, monkeypatch):
    labtasker = pytest.importorskip("labtasker")

    path = tmp_path / "progress.json"
    reports = []
    monkeypatch.setattr(labtasker, "report_progress", lambda value: reports.append(value) is None)
    reporter = ProgressFileReporter(path, base={"operation": "run_eval", "task": "pick_cube"})
    assert not reporter.poll()
    write_progress(path, {"completed": 1, "total": 3, "successes": 1})
    assert reporter.poll()
    assert not reporter.poll()
    write_progress(path, {"completed": 2, "total": 3, "successes": 1})
    assert reporter.poll()
    assert [report["completed"] for report in reports] == [1, 2]


def test_progress_file_reporter_allows_environment_without_labtasker(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    write_progress(path, {"completed": 1, "total": 1})
    reporter = ProgressFileReporter(path, base={"operation": "run_eval"})
    original_import = builtins.__import__

    def import_without_labtasker(name, *args, **kwargs):
        if name == "labtasker":
            raise ImportError("not installed in evaluator environment")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_labtasker)
    assert reporter.poll()
    assert not reporter.poll()
