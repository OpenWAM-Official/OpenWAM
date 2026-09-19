"""RoboTwin sealed instruction and absolute episode behavior."""

import json
import random
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("labtasker")

from benchmarks.robotwin import eval_policy_wrapper
from benchmarks.robotwin import labtasker_runtime as rt

M = SimpleNamespace(eval_policy_wrapper=eval_policy_wrapper)


def test_result_validation_rejects_wrong_seed_or_episode(tmp_path):
    from benchmarks.utils.eval_manifest import seal_manifest, write_manifest

    manifest = seal_manifest(
        {
            "benchmark": "robotwin",
            "task": "adjust_bottle",
            "mode": "demo_clean",
            "entries": [
                {
                    "episode": index,
                    "seed": 100000 + index,
                    "instructions": {"seen": f"seen-{index}", "unseen": f"instruction-{index}"},
                }
                for index in range(10)
            ],
        }
    )
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    inputs = {
        "operation": "run_eval",
        "task": "adjust_bottle",
        "mode": "demo_clean",
        "instruction_type": "unseen",
        "episode_start": 3,
        "num_episodes": 2,
        "total_episodes": 5,
        "manifest_hash": manifest["manifest_hash"],
        "manifest_path": str(manifest_path),
    }
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "operation": "run_eval",
                "task": "adjust_bottle",
                "mode": "demo_clean",
                "instruction_type": "unseen",
                "episode_start": 3,
                "num_episodes": 2,
                "total_episodes": 5,
                "successes": 2,
                "episodes": [
                    {
                        "episode": 3,
                        "seed": 100003,
                        "instruction": "instruction-3",
                        "success": True,
                    },
                    {
                        "episode": 4,
                        "seed": 999,
                        "instruction": "instruction-4",
                        "success": True,
                    },
                ],
                "manifest_hash": manifest["manifest_hash"],
            }
        )
    )
    with pytest.raises(ValueError, match="inconsistent"):
        rt.read_client_result(path, inputs)


def test_seed_planner_preserves_skipped_seed_order(tmp_path, monkeypatch):
    class Env:
        def setup_demo(self, now_ep_num, seed, **kwargs):
            self.seed = seed

        def play_once(self):
            self.plan_success = self.seed % 2 == 0
            return {"info": {"seed": self.seed}}

        def check_success(self):
            return True

        def close_env(self):
            pass

    module = SimpleNamespace(
        eval_function_decorator=lambda *_: None,
        eval_policy=lambda *_: None,
        generate_episode_descriptions=lambda task, infos, limit: [
            {kind: [f"instruction-{infos[0]['seed']}"] for kind in ("seen", "unseen")}
        ],
    )
    result = tmp_path / "manifest.json"
    progress = tmp_path / "progress.json"
    monkeypatch.setenv("ROBOTWIN_LABTASKER_RESULT", str(result))
    monkeypatch.setenv("ROBOTWIN_PROGRESS_FILE", str(progress))
    M.eval_policy_wrapper._install_manifest(module)
    next_seed, successes = module.eval_policy(
        "adjust_bottle", Env(), {"render_freq": 1, "task_config": "demo_clean"}, None, 100000, test_num=3
    )
    payload = json.loads(result.read_text())
    assert [entry["seed"] for entry in payload["entries"]] == [100000, 100002, 100004]
    assert [entry["instructions"]["unseen"] for entry in payload["entries"]] == [
        "instruction-100000",
        "instruction-100002",
        "instruction-100004",
    ]
    assert (next_seed, successes) == (100005, 0)
    assert json.loads(progress.read_text())["completed"] == 3


def test_seed_planner_isolated_from_caller_rng(tmp_path, monkeypatch):
    class Env:
        def setup_demo(self, now_ep_num, seed, **kwargs):
            self.seed = seed
            self.sample = random.random()

        def play_once(self):
            self.plan_success = random.random() > 0.25
            return {"info": {"seed": self.seed, "sample": self.sample, "draw": random.random()}}

        def check_success(self):
            return True

        def close_env(self):
            random.random()

    def build(index):
        module = SimpleNamespace(
            eval_function_decorator=lambda *_: None,
            eval_policy=lambda *_: None,
            generate_episode_descriptions=lambda task, infos, limit: [
                {kind: [f"{task}-{infos[0]['seed']}-a", f"{task}-{infos[0]['seed']}-b"] for kind in ("seen", "unseen")}
            ],
        )
        result = tmp_path / f"manifest-{index}.json"
        monkeypatch.setenv("ROBOTWIN_LABTASKER_RESULT", str(result))
        monkeypatch.setenv("ROBOTWIN_PROGRESS_FILE", str(tmp_path / f"progress-{index}.json"))
        M.eval_policy_wrapper._install_manifest(module)
        module.eval_policy(
            "adjust_bottle",
            Env(),
            {"render_freq": 1, "task_config": "demo_clean"},
            None,
            100000,
            test_num=5,
        )
        return json.loads(result.read_text())

    random.seed(1)
    first = build(1)
    random.seed(987654321)
    second = build(2)
    assert first == second
    assert first["manifest_hash"] == second["manifest_hash"]


def test_eval_task_state_matches_robotwin_play_once_assignments(monkeypatch):
    import types

    class ArmTag(str):
        pass

    def make_env(task, *, object_position=None, laptop_quaternion=None, face_prod=None):
        module_name = f"fake_robotwin_env_{task}"
        task_module = types.ModuleType(module_name)
        task_module.ArmTag = ArmTag
        if face_prod is not None:
            task_module.get_face_prod = lambda *_: face_prod
        monkeypatch.setitem(sys.modules, module_name, task_module)

        env_type = type("Env", (), {"__module__": module_name})
        env = env_type()
        if object_position is not None:
            env.object = SimpleNamespace(get_pose=lambda: SimpleNamespace(p=object_position))
        if laptop_quaternion is not None:
            env.laptop = SimpleNamespace(get_pose=lambda: SimpleNamespace(q=laptop_quaternion))
        return env

    laptop = make_env("open_laptop", laptop_quaternion=[1, 0, 0, 0], face_prod=1)
    M.eval_policy_wrapper._initialize_eval_task_state("open_laptop", laptop)
    assert laptop.arm_tag == ArmTag("left")

    scale = make_env("place_object_scale", object_position=[0.2, 0.0, 0.4])
    M.eval_policy_wrapper._initialize_eval_task_state("place_object_scale", scale)
    assert scale.arm_tag == ArmTag("right")

    cabinet = make_env("put_object_cabinet", object_position=[-0.2, 0.0, 0.37])
    M.eval_policy_wrapper._initialize_eval_task_state("put_object_cabinet", cabinet)
    assert cabinet.arm_tag == ArmTag("left")
    assert cabinet.origin_z == 0.37


def test_eval_uses_planned_seeds_and_absolute_indices(tmp_path, monkeypatch):
    import types

    from benchmarks.utils.eval_manifest import seal_manifest, write_manifest

    manifest = seal_manifest(
        {
            "benchmark": "robotwin",
            "task": "adjust_bottle",
            "mode": "demo_clean",
            "entries": [
                {
                    "episode": index,
                    "seed": 100000 + index,
                    "instructions": {"seen": f"seen-{index}", "unseen": f"instruction-{index}"},
                }
                for index in range(10)
            ],
        }
    )
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    policy = types.ModuleType("fake_robotwin_policy")
    policy.set_benchmark_rng_domain = lambda domain: None
    monkeypatch.setitem(sys.modules, policy.__name__, policy)

    def decorator(_policy_name, name):
        if name == "reset_model":
            return lambda model: None
        if name == "eval":

            def evaluate(env, model, observation):
                env.take_action_cnt += 1
                env.eval_success = env.seed % 2 == 0

            return evaluate
        raise AssertionError(name)

    module = SimpleNamespace(eval_function_decorator=decorator)
    result = tmp_path / "batch.json"
    progress = tmp_path / "progress.json"
    monkeypatch.setenv("ROBOTWIN_LABTASKER_RESULT", str(result))
    monkeypatch.setenv("ROBOTWIN_PROGRESS_FILE", str(progress))
    monkeypatch.setenv("ROBOTWIN_MANIFEST_HASH", manifest["manifest_hash"])
    monkeypatch.setenv("ROBOTWIN_EPISODE_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ROBOTWIN_EPISODE_START", "4")
    monkeypatch.setenv("ROBOTWIN_EPISODE_COUNT", "2")
    monkeypatch.setenv("ROBOTWIN_TOTAL_EPISODES", "10")
    monkeypatch.setenv("ROBOTWIN_INSTRUCTION_TYPE", "unseen")
    M.eval_policy_wrapper._install_eval(module)
    cuda_seeds = []
    initialized = []
    monkeypatch.setattr(M.eval_policy_wrapper, "_canonicalize_torch_cuda", cuda_seeds.append)
    monkeypatch.setattr(
        M.eval_policy_wrapper,
        "_initialize_eval_task_state",
        lambda task, env: initialized.append((task, env.seed)),
    )

    class Env:
        def __init__(self):
            self.calls = []
            self.eval_video_path = None
            self.render_freq = 0

        def setup_demo(self, now_ep_num, seed, **kwargs):
            self.calls.append((now_ep_num, seed))
            self.seed = seed
            self.take_action_cnt = 0
            self.step_lim = 1
            self.eval_success = False

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_obs(self):
            return {}

        def close_env(self, **kwargs):
            pass

    env = Env()
    next_seed, successes = module.eval_policy(
        "adjust_bottle",
        env,
        {
            "task_config": "demo_clean",
            "policy_name": policy.__name__,
            "clear_cache_freq": 10,
        },
        None,
        0,
        test_num=2,
    )
    payload = json.loads(result.read_text())
    assert env.calls == [(4, 100004), (5, 100005)]
    assert initialized == [("adjust_bottle", 100004), ("adjust_bottle", 100005)]
    assert cuda_seeds == [100004, 100005]
    assert [row["instruction"] for row in payload["episodes"]] == ["instruction-4", "instruction-5"]
    assert [row["success"] for row in payload["episodes"]] == [True, False]
    assert (next_seed, successes) == (100006, 1)
    assert json.loads(progress.read_text()) == {
        "completed": 2,
        "episode": 5,
        "successes": 1,
        "total": 2,
    }
