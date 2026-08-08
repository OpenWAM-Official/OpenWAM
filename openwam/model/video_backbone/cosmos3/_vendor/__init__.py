"""Vendored upstream modeling code for the Cosmos3 backbone.

Import nothing from here at module scope elsewhere in openwam — the vendored
file imports ``diffusers`` eagerly, so it must only be reached through
``from_pretrained``-time lazy imports (CPU-CI safety). See README.md in this
directory for provenance and the local-modification policy.
"""
