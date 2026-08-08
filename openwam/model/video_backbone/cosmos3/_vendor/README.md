# Vendored Cosmos3 modeling code

| file | upstream path | source commit |
|---|---|---|
| `transformer_cosmos3.py` | `src/diffusers/models/transformers/transformer_cosmos3.py` | huggingface/diffusers `6ad357395d936c4d27347463f938cdbb400a6e59` |

## Why vendored

`Cosmos3OmniTransformer` with Cosmos3-**Edge** support (relu² / nemotron RMSNorm /
`k_norm_und_for_gen`, fixed in PR #14246) exists only on diffusers `main` as of
2026-08; PyPI 0.39.0 ships base Cosmos3 without these and would silently build a
wrong model for the Edge checkpoint. The file's eight diffusers-internal imports
(`ConfigMixin`, `PeftAdapterMixin`, `BaseOutput`, `AttentionMixin`,
`AttentionModuleMixin`, `dispatch_attention_fn` incl. `enable_gqa`,
`TimestepEmbedding`/`Timesteps`, `ModelMixin`, `RMSNorm`) exist unchanged in
diffusers 0.38/0.39, so vendoring this single file lets the training venv
(diffusers 0.38.0) run it without any dependency upgrade.

## Local modifications

Exactly one: the package-relative imports (`from ...configuration_utils import …`)
were rewritten to absolute `diffusers.*` imports, plus the provenance note in the
header. Everything else is byte-identical to upstream. Keep it that way — put any
OpenWAM-side logic in `openwam/model/video_backbone/cosmos3/` proper, never here,
so future upstream diffs stay one-line.

## Retirement plan

When a stable diffusers release (expected 0.40, ~2026-09) ships the Edge-complete
`Cosmos3OmniTransformer`, delete this directory and import from diffusers instead
(also re-check open PR #14272 — Edge UniPC scheduler + framework-matching i2v
preprocessing — for the deploy path).

License: Apache-2.0 (header retained in the file). Copyright NVIDIA Team and
HuggingFace Team.
