# Third-Party Notices

OpenWAM includes or adapts source code from the projects listed below. These
notices apply to the source code in this repository; model weights, datasets,
and other separately distributed assets may have different license terms.

The repository-level OpenWAM license does not replace the licenses identified
here. Full license texts are included under `third_party/licenses/`.

## FastWAM

- Source: <https://github.com/yuantianyuan01/FastWAM>
- Upstream revision: **UNKNOWN** (the original internal import did not record a
  commit SHA).
- License: MIT; see `third_party/licenses/FastWAM-MIT.txt`.
- OpenWAM use: adaptations and design-derived implementation in the joint
  video/action attention, action backbone, conditioning, and multiview data
  paths.
- Modifications: reorganized around OpenWAM's backbone and architecture
  interfaces, extended mask and conditioning behavior, and integrated with the
  OpenWAM training and evaluation stack.

## ImageWAM

- Source: <https://github.com/yuyangalin/ImageWAM>
- Upstream revision: **UNKNOWN** (the original internal import did not record a
  commit SHA).
- License: MIT; see `third_party/licenses/ImageWAM-MIT.txt`.
- OpenWAM use: the per-suite LIBERO task-sampling behavior in
  `benchmarks/libero/scheduler.py`.
- Modifications: integrated into OpenWAM's shared benchmark scheduler and
  deterministic job planning.

## V-JEPA 2

- Source: <https://github.com/facebookresearch/vjepa2>
- Upstream revision: `ce64921e94f0ffdc330c00fc62618157894b74be`.
- Copyright: Meta Platforms, Inc. and affiliates.
- License: MIT; see `third_party/licenses/V-JEPA2-MIT.txt`.
- OpenWAM use: vendored and adapted V-JEPA ViT components under
  `openwam/model/video_backbone/encoder/vjepa21_src/`.
- Modifications: packaged in-tree, coupled to an OpenWAM manifest/weight loader,
  and patched for OpenWAM's dtype and encoder contracts.

## FLUX.2 inference source

- Source: <https://github.com/black-forest-labs/flux2>
- Upstream revision: **UNKNOWN** (the original internal import did not record a
  commit SHA).
- Copyright: Black Forest Labs Inc.
- License for the upstream inference source: Apache License 2.0; see
  `third_party/licenses/Apache-2.0.txt`.
- OpenWAM use: encoder-side autoencoder implementation in
  `openwam/model/video_backbone/encoder/flux2_vae_src/autoencoder.py`.
- Modifications: removed decoder-only functionality and added OpenWAM-specific
  checkpoint conversion and encoder integration.

This notice covers source code only. FLUX model weights, including FLUX.2-dev
weights, are distributed separately and are subject to their own model license.

## Motus

- Source: <https://github.com/thu-ml/Motus>
- Upstream revision: **UNKNOWN** (the original internal import did not record a
  commit SHA).
- License: Apache License 2.0; see
  `third_party/licenses/Apache-2.0.txt`.
- OpenWAM use: adapted trimodal Mixture-of-Transformers and understanding-expert
  implementation under `openwam/model/architectures/tri_system/`.
- Modifications: restructured for OpenWAM's modular backbones, attention-mask
  contracts, checkpointing, configuration, and training lifecycle.

## CameraCtrl

- Source: <https://github.com/hehao13/CameraCtrl>
- Upstream revision: **UNKNOWN** (the original internal import did not record a
  commit SHA).
- Copyright: the CameraCtrl authors.
- License: Apache License 2.0; see
  `third_party/licenses/Apache-2.0.txt`.
- OpenWAM use: camera-pose and ray-conditioning utilities in
  `openwam/model/video_backbone/wan/camera_controller.py`.
- Modifications: integrated with OpenWAM's Wan backbone and adapter interface.

Additional Wan-backbone attribution is retained in
`openwam/model/video_backbone/wan/license/NOTICE`.

## DiffSynth-Studio / Wan backbone

- Source: <https://github.com/modelscope/DiffSynth-Studio>
- Upstream revision: **UNKNOWN** (the original internal import did not record a
  commit SHA).
- License: Apache License 2.0; see
  `third_party/licenses/Apache-2.0.txt`.
- OpenWAM use: extracted and adapted Wan model loading, model components,
  conditioning, diffusion scheduling, state-dict conversion, and VRAM helpers
  under `openwam/model/video_backbone/wan/`.
- Modifications: removed the former vendored pipeline boundary, reorganized the
  implementation as an OpenWAM backbone, added training/deployment integration,
  and extended support for Wan variants.

The upstream source license does not govern separately downloaded Wan model
weights or other model assets. Their distributors' terms apply separately.

## Hugging Face diffusers (Cosmos3 transformer)

- Source: <https://github.com/huggingface/diffusers>
- Upstream revision: `6ad357395d936c4d27347463f938cdbb400a6e59`.
- Copyright: 2025 The NVIDIA Team and The HuggingFace Team.
- License: Apache License 2.0; see
  `third_party/licenses/Apache-2.0.txt`.
- OpenWAM use: vendored `Cosmos3OmniTransformer` implementation in
  `openwam/model/video_backbone/cosmos3/_vendor/transformer_cosmos3.py`.
- Modifications: package-relative diffusers imports were changed to absolute
  imports so the file can be used with OpenWAM's installed diffusers package.
  Detailed provenance is retained in the adjacent `_vendor/README.md`.

## NVIDIA Cosmos-Predict2.5

- Source: <https://github.com/nvidia-cosmos/cosmos-predict2.5>
- Upstream revision: `441b89740d91922737008a61e7f71407d47944e7`
  (the gitlink recorded at `third_party/cosmos-predict2.5`).
- Copyright: NVIDIA Corporation and affiliates.
- License for the upstream source code: Apache License 2.0; see
  `third_party/licenses/Apache-2.0.txt`.
- OpenWAM use: an external git submodule plus OpenWAM-side adapters under
  `openwam/model/video_backbone/cosmos_predict25/`.
- Modifications: OpenWAM installs the pinned upstream package and integrates it
  behind OpenWAM's backbone, checkpoint, and deployment interfaces; the
  submodule itself remains a separately versioned upstream checkout.

This notice covers source code only. Cosmos model weights are distributed
separately under NVIDIA's applicable model terms, not the Apache-2.0 source
license.

## LIBERO

- Source: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- Upstream revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Copyright: 2023 Lifelong Robot Learning.
- License: MIT; see `third_party/licenses/LIBERO-MIT.txt`.
- OpenWAM use: the reproducible external benchmark checkout configured by
  `benchmarks/libero/setup_env.sh`; the upstream checkout is not vendored in
  this repository.
- Modifications: `benchmarks/libero/patches/libero-pytorch-load.patch` makes
  the pinned checkout explicit about loading its trusted NumPy-containing
  init-state assets under newer PyTorch releases.
