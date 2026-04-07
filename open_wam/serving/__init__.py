"""Serving utilities for deployment workflows."""

__all__ = ["PolicyServer"]


def __getattr__(name):
    if name == "PolicyServer":
        from open_wam.serving.policy_server import PolicyServer

        return PolicyServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
