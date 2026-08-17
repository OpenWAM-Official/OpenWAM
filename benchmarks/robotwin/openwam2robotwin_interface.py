"""RoboTwin eval adapter for the OpenWAM Policy Server.

Loaded by RoboTwin's ``eval_policy.py`` via ``--policy_name``. It
communicates with a running OpenWAM WebSocket server instead of loading model
weights directly, so the RoboTwin client environment only needs::

    numpy, opencv-python, Pillow   (see requirements.txt)

Protocol overview (WebSocket):

    obs   — send 3 camera lossless PNGs + task prompt, get action vector
    reset — clear server episode state before a new rollout
    ping  — liveness probe

Server default port: 8848.

Camera mapping from RoboTwin to the OpenWAM server's fixed client API names:

    RoboTwin              →  OpenWAM client field
    head_camera           →  head_camera        (required)
    left_camera           →  left_wrist_camera  (optional)
    right_camera          →  right_wrist_camera (optional)
    front_camera          →  dropped (not part of the OpenWAM contract)

All image preprocessing (resize, multi-view composition) happens server-side,
driven by the checkpoint's saved ``config.yaml``. The server is prompt-agnostic:
it forwards the prompt to the model verbatim, so this adapter wraps the raw task
instruction with RoboTwin's own deploy template (``prompt_template``) before
sending. Camera frames are streamed raw.
"""

# benchmarks.utils lives one level up. single_eval.sh only puts benchmarks/robotwin/
# on PYTHONPATH, so add the project root here to make the shared helpers
# (payload assembly, WebSocket client, action conversions) importable from the
# RoboTwin eval process too.
import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, Optional  # noqa: E402

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from benchmarks.robotwin.prompt_template import format_prompt_for_inference  # noqa: E402
from benchmarks.utils import WSPolicyClient, action_conversion, client, transport  # noqa: E402

# --- Per-task step_lim overrides ---
# A single YAML file of {task_name: int} lets users override RoboTwin's
# upstream task_config/_eval_step_limit.yml without touching the RoboTwin
# source tree. Missing tasks keep their upstream value (RoboTwin falls back
# to 1000 when neither side defines one).
_STEP_LIMITS_PATH = os.path.join(os.path.dirname(__file__), "step_limits.yml")


def _load_step_lim_overrides(path: str) -> Dict[str, int]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as exc:
        print(f"[OpenWAMClient] Failed to load step_lim overrides from {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[OpenWAMClient] {path} must be a task_name->int mapping; ignoring.")
        return {}
    out: Dict[str, int] = {}
    for k, v in data.items():
        # Reject bool (which is an int subclass in Python) and non-int numerics
        # explicitly — silent ``int(True) == 1`` or ``int(160.9) == 160`` caps
        # would be nearly impossible to debug from a one-line override log.
        if isinstance(v, bool) or not isinstance(v, int):
            print(
                f"[OpenWAMClient] Skipping step_lim override {k!r}={v!r}: "
                f"value must be a plain int (got {type(v).__name__})"
            )
            continue
        out[str(k)] = v
    return out


_STEP_LIM_OVERRIDES: Dict[str, int] = _load_step_lim_overrides(_STEP_LIMITS_PATH)
_MISSING_TASK_NAME_WARNED: bool = False
# RoboTwin resets ``TASK_ENV.step_lim`` from its own upstream YAML at the start
# of every episode, so a naive ``prev != override`` check would re-log on every
# episode. Track the ``(task_name, override)`` pairs we have already announced
# and stay silent for the rest of the process.
_LOGGED_OVERRIDES: set = set()


def _parse_bool(value, default: bool = False) -> bool:
    """Parse YAML/CLI boolean values without treating "false" as truthy."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off", "none", "null", ""):
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _parse_optional_int(value, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer or null, got {value!r}")
    return parsed


def _apply_step_lim_override(task_env) -> None:
    # First step of each episode: apply per-task step_lim override (if any).
    # The RoboTwin eval loop re-checks ``task_env.step_lim`` on every iteration,
    # so mutating it here takes effect from the next iteration onward.
    if not _STEP_LIM_OVERRIDES or getattr(task_env, "take_action_cnt", -1) != 0:
        return
    task_name = getattr(task_env, "task_name", None)
    if not task_name:
        global _MISSING_TASK_NAME_WARNED
        if not _MISSING_TASK_NAME_WARNED:
            _MISSING_TASK_NAME_WARNED = True
            print(
                "[OpenWAMClient] step_lim overrides loaded but TASK_ENV.task_name "
                f"is missing/empty ({task_name!r}); overrides will not be applied."
            )
        return
    override = _STEP_LIM_OVERRIDES.get(task_name)
    if override is None or getattr(task_env, "step_lim", None) == override:
        return
    prev = getattr(task_env, "step_lim", None)
    task_env.step_lim = override
    key = (task_name, override)
    if key not in _LOGGED_OVERRIDES:
        _LOGGED_OVERRIDES.add(key)
        print(f"[OpenWAMClient] step_lim override: {task_name} {prev} -> {override}")


class ModelClient:
    """RoboTwin ``ModelClient`` backed by the OpenWAM Policy Server.

    The server manages action chunking internally, so this client sends an obs
    message every step and lets the server decide whether to run full diffusion
    inference or pop a cached action from its buffer.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        send_state: bool = True,
        state_dim: Optional[int] = None,
        request_timeout: int = 300,
        action_indices: Optional[list] = None,
        action_type: str = "qpos",
        debug: bool = False,
        debug_dir: str = "./debug_images",
    ) -> None:
        """
        Args:
            host:            OpenWAM server hostname / IP.
            port:            OpenWAM WebSocket port (default 8848).
            send_state:      Whether to include the robot proprioceptive state
                             vector in the obs message.
            state_dim:       Optional expected proprio dimension. When set, the
                             client fails fast if the extracted state does not
                             match the checkpoint's architecture.state_dim.
            request_timeout: WebSocket timeout in seconds.
            action_indices:  Optional index list to reorder the returned action
                             vector before passing it to the environment.
                             None = no reordering.
            action_type:     How the returned action is passed to ``take_action``.
                             ``'qpos'`` means 14D joint angles straight through;
                             ``'ee'`` converts the server's 20D EEF output to
                             the 16D end-effector action expected by RoboTwin.
            debug:           Save per-step images + JSON metadata under debug_dir.
            debug_dir:       Root directory for debug artifacts.
        """
        _VALID_ACTION_TYPES = ("qpos", "ee")
        if action_type not in _VALID_ACTION_TYPES:
            raise ValueError(
                f"[OpenWAMClient] Unsupported action_type={action_type!r}; "
                f"expected one of {_VALID_ACTION_TYPES}. "
                f"Note: EEF-mode training uses 'ee' (NOT 'eef')."
            )

        self._send_state = send_state
        self._state_dim = state_dim
        self._request_timeout = request_timeout
        self._action_indices = action_indices
        self._action_type = action_type
        self._task_description = ""
        self._debug = debug
        self._debug_dir = debug_dir
        self._episode = -1  # incremented to 0 on the first reset_model() call
        self._step = 0

        self._ws_url = f"ws://{host}:{port}"
        self._client = WSPolicyClient(self._ws_url, timeout=request_timeout)

        print(
            f"[OpenWAMClient] server={self._ws_url} send_state={send_state} "
            f"state_dim={state_dim} "
            f"request_timeout={request_timeout}s action_type={action_type} "
            f"action_indices={action_indices} debug={debug} debug_dir={debug_dir}"
        )

        self._wait_until_healthy()

    def _wait_until_healthy(self, timeout_s: int = 300, poll_interval: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_exc: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                if self._client.ping().get("type") == transport.PONG:
                    print(f"[OpenWAMClient] Server healthy at {self._ws_url}")
                    return
            except Exception as exc:
                last_exc = exc
                self._client.close()  # drop the half-open socket before retrying
            time.sleep(poll_interval)
        raise RuntimeError(
            f"OpenWAM server did not become healthy within {timeout_s}s at {self._ws_url}. Last error: {last_exc}"
        )

    def reset(self, task_description: str = "") -> None:
        """Clear server episode state and (optionally) bump the debug episode counter.

        RoboTwin's ``reset_model()`` passes ``task_description=""`` at episode
        boundaries. We also reset internally when the task instruction changes
        mid-rollout, in which case ``task_description`` is non-empty.
        """
        if task_description == "":
            # Episode boundary — create a fresh debug dir.
            self._episode += 1
            self._step = 0
            if self._debug:
                ep_dir = os.path.join(self._debug_dir, f"ep{self._episode:04d}")
                os.makedirs(ep_dir, exist_ok=True)
                print(f"[OpenWAMClient] debug images → {ep_dir}")
        self._task_description = task_description
        result = self._client.reset()
        if result.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"[OpenWAMClient] Server reset failed: {result}")

    def _save_debug_step(
        self,
        cams: dict,
        prompt: str,
        state_list: Optional[list],
        response: dict,
    ) -> None:
        """Save all per-step debug data under debug_dir/ep{N:04d}/step_{N:04d}/.

        Files written:
          head.png / left.png / right.png  — raw per-camera frames as client sent
                                             (missing wrist → *_missing.txt stub)
          meta.json                        — prompt, state, action, latency, step, episode
        """
        ep_dir = os.path.join(self._debug_dir, f"ep{self._episode:04d}")
        step_dir = os.path.join(ep_dir, f"step_{self._step:04d}")
        os.makedirs(step_dir, exist_ok=True)

        for name in ("head", "left", "right"):
            img = cams.get(name)
            if img is None:
                with open(os.path.join(step_dir, f"{name}_missing.txt"), "w") as f:
                    f.write("client sent None for this camera\n")
            else:
                cv.imwrite(os.path.join(step_dir, f"{name}.png"), cv.cvtColor(img, cv.COLOR_RGB2BGR))

        meta = {
            "episode": self._episode,
            "step": self._step,
            "prompt": prompt,
            "state": state_list,
            "action": response.get("action"),
            "server_step": response.get("step"),
            "latency_ms": response.get("latency_ms"),
        }
        with open(os.path.join(step_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def step(self, example: dict) -> np.ndarray:
        """
        Submit one observation to the server and return the next action.

        Args:
            example: {
                "cams": {
                    "head":  np.ndarray,      # HxWx3 RGB uint8, required
                    "left":  np.ndarray|None, # HxWx3 RGB uint8, optional
                    "right": np.ndarray|None, # HxWx3 RGB uint8, optional
                },
                "lang":  str,                 # task instruction
                "state": np.ndarray,          # proprioceptive state (optional)
            }

        Returns:
            action: np.ndarray, shape (action_dim,)
        """
        cams = example["cams"]
        instruction = str(example.get("lang", self._task_description))

        # Mirror the upstream pattern: reset if the task instruction changes.
        if instruction and instruction != self._task_description:
            self.reset(instruction)

        state_arr = example.get("state", None)
        state_list: Optional[list] = None
        if self._send_state:
            if state_arr is None:
                raise ValueError(
                    "[OpenWAMClient] send_state=True requires example['state']. "
                    "New proprio-conditioned checkpoints need this field; set "
                    "send_state=false only for checkpoints trained without state."
                )
            state_np = np.asarray(state_arr, dtype=np.float32).reshape(-1)
            if self._state_dim is not None and state_np.size != self._state_dim:
                raise ValueError(
                    f"[OpenWAMClient] Extracted state_dim={state_np.size}, expected {self._state_dim}. "
                    "Check policy_config.yml: action_type/state_dim must match the checkpoint's "
                    "dataloader.action_mode and architecture.state_dim."
                )
            state_list = [float(v) for v in state_np]

        # Server is prompt-agnostic; RoboTwin wraps its own instruction here (see prompt_template).
        prompt = format_prompt_for_inference(instruction)
        payload = client.build_payload(
            head=client.encode_numpy_b64(cams["head"]),
            left_wrist=client.encode_numpy_b64(cams["left"]) if cams.get("left") is not None else None,
            right_wrist=client.encode_numpy_b64(cams["right"]) if cams.get("right") is not None else None,
            prompt=prompt,
            state=state_list,
        )
        response = self._client.predict(payload)
        action = np.array(response["action"], dtype=np.float32)

        self._step += 1
        if self._debug:
            self._save_debug_step(cams, prompt, state_list, response)

        if self._action_indices is not None:
            action = action[self._action_indices]

        return action


# ---------------------------------------------------------------------------
# Module-level entry points called by RoboTwin's eval_policy.py
# ---------------------------------------------------------------------------


def get_model(usr_args: dict) -> ModelClient:
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 8848)),
        send_state=_parse_bool(usr_args.get("send_state", True), default=True),
        state_dim=_parse_optional_int(usr_args.get("state_dim", None), "state_dim"),
        request_timeout=int(usr_args.get("request_timeout", 300)),
        action_indices=usr_args.get("action_indices", None),
        action_type=usr_args.get("action_type", "qpos"),
        debug=_parse_bool(usr_args.get("debug", False), default=False),
        debug_dir=usr_args.get("debug_dir", "./debug_images"),
    )


def reset_model(model: ModelClient) -> None:
    model.reset(task_description="")


def _extract_eef_proprio(observation: dict) -> np.ndarray:
    endpose = observation.get("endpose")
    if not isinstance(endpose, dict):
        available = ", ".join(sorted(observation.keys()))
        raise KeyError(
            "action_type='ee' requires RoboTwin endpose proprio matching training action_mode='eef'. "
            "Expected observation['endpose'] with left_endpose, right_endpose, left_gripper, right_gripper. "
            f"Available top-level observation keys: {available}"
        )

    return action_conversion.robotwin_endpose_to_eef20d(
        endpose["left_endpose"],
        endpose["right_endpose"],
        endpose["left_gripper"],
        endpose["right_gripper"],
    )


def _extract_proprio(model: ModelClient, observation: dict) -> np.ndarray:
    if model._action_type == "ee":
        return _extract_eef_proprio(observation)
    try:
        return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    except KeyError as exc:
        available = ", ".join(sorted(observation.keys()))
        raise KeyError(
            "action_type='qpos' requires RoboTwin joint proprio matching training action_mode='joint'. "
            "Expected observation['joint_action']['vector']. "
            f"Available top-level observation keys: {available}"
        ) from exc


def eval(TASK_ENV, model: ModelClient, observation: dict) -> None:
    """Per-step callback invoked by RoboTwin's eval_policy.py.

    RoboTwin exposes three per-camera entries under ``observation["observation"]``
    (``head_camera`` / ``left_camera`` / ``right_camera``). They map positionally
    to the OpenWAM client API's fixed fields (head / left_wrist / right_wrist).
    ``front_camera`` is ignored — it's not part of the server contract.
    """
    _apply_step_lim_override(TASK_ENV)

    instruction = TASK_ENV.get_instruction()
    obs = observation["observation"]

    example = {
        "cams": {
            "head": obs["head_camera"]["rgb"],
            "left": obs.get("left_camera", {}).get("rgb"),
            "right": obs.get("right_camera", {}).get("rgb"),
        },
        "lang": str(instruction),
        "state": _extract_proprio(model, observation) if model._send_state else None,
    }

    action = model.step(example)

    # EEF mode: convert 20D (xyz+rot6d+grip)×2 → 16D (xyz+quat+grip)×2.
    if model._action_type == "ee" and len(action) == 20:
        action = action_conversion.eef20d_to_ee16d(action)

    TASK_ENV.take_action(action, action_type=model._action_type)
