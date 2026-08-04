"""RoboCasa365-specific deploy prompt template.

The OpenWAM policy server is *prompt-agnostic*: it forwards whatever ``prompt``
a client sends straight to the model (see ``openwam/deploy/obs_preprocess.py``).
Each benchmark therefore owns the prompt processing its checkpoints were trained
with. This module is RoboCasa365's: it wraps the env's raw task instruction
(``annotation.human.task_description``) in the exact template the robocasa365
dataloader applies at training time, so eval-time prompts stay in-distribution.

The wrapped string must stay byte-for-byte identical to
``openwam.dataloader.transforms.multiview.format_prompt_for_inference`` — the two
are the eval and training ends of the same contract. A regression test
(``tests/test_robocasa365_config.py``) pins them together, mirroring robotwin's
``benchmarks/robotwin/prompt_template.py``.

Deliberately dependency-free (no ``openwam`` / torch imports) so it loads inside
the thin RoboCasa365 eval environment.
"""

# Byte-for-byte identical to the training-time wrapper in
# openwam/dataloader/transforms/multiview.py. Kept as a bare prefix + concat (not
# str.format) so equality with the training template is obvious by construction.
_DEPLOY_PROMPT_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "


def format_prompt_for_inference(base_prompt: str) -> str:
    """Wrap a raw RoboCasa365 instruction in the training-time deploy template."""
    return _DEPLOY_PROMPT_PREFIX + base_prompt


__all__ = ["format_prompt_for_inference"]
