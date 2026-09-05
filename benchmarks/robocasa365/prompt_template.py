"""Canonical RoboCasa365 deploy prompt handling.

The OpenWAM policy server is *prompt-agnostic*: it forwards whatever ``prompt``
a client sends straight to the model (see ``openwam/deploy/obs_preprocess.py``).
RoboCasa365 deliberately uses the raw environment task instruction
(``annotation.human.task_description``), matching the native task text stored in
the converted LeRobot dataset. No benchmark-specific prefix or suffix is added.

Deliberately dependency-free (no ``openwam`` / torch imports) so it loads inside
the thin RoboCasa365 eval environment.
"""


def format_prompt_for_inference(base_prompt: str) -> str:
    """Return the native RoboCasa365 instruction unchanged."""
    return base_prompt


__all__ = ["format_prompt_for_inference"]
