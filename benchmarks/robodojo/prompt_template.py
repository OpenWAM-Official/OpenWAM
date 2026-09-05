"""RoboDojo deploy prompt template.

The OpenWAM policy server is prompt-agnostic: it forwards whatever ``prompt``
a client sends straight to the model. This wrapper must stay byte-for-byte
identical to
``openwam.dataloader.transforms.multiview.format_prompt_for_inference``.
``tests/test_robodojo_runtime_boundary.py`` pins the two ends together.

Deliberately dependency-free so it loads in the Isaac eval environment.
"""

_DEPLOY_PROMPT_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "


def format_prompt_for_inference(base_prompt: str) -> str:
    """Wrap a raw RoboDojo instruction in the training-time deploy template."""
    return _DEPLOY_PROMPT_PREFIX + base_prompt


__all__ = ["format_prompt_for_inference"]
