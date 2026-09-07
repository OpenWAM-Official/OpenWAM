"""EBench deploy prompt wrapper.

The prefix must match the training-time template byte-for-byte. Keep this
module dependency-free so it can run in the GenManip client environment.
"""

_DEPLOY_PROMPT_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "


def format_prompt_for_inference(base_prompt: str) -> str:
    """Wrap a raw EBench instruction in the training-time deploy template."""
    return _DEPLOY_PROMPT_PREFIX + base_prompt


__all__ = ["format_prompt_for_inference"]
