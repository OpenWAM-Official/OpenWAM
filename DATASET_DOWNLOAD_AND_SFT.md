# H200 数据下载与 RoboCasa GR1 SFT 一键配置

## 1. 设置存储目录

只需修改 `DATA_ROOT`：

```bash
export DATA_ROOT=/path/to/h200/storage/datasets
mkdir -p "${DATA_ROOT}"
```

建议目录：

```text
${DATA_ROOT}/
├── libero-lerobot-v3/
├── robocasa-gr1-24k/          # NVIDIA LeRobot v2.0 buckets
├── robocasa-gr1-eef-v20/      # trusted EEF-enriched source
├── robocasa-gr1-eef-v30/      # OpenWAM v3 re-index
└── robocasa-gr1-tabletop-tasks/
```

## 2. 安装下载工具

```bash
python -m pip install -U "huggingface_hub[cli]"
hf auth login
```

公开数据通常不要求登录，但登录后限流更宽。

## 3. 一键下载

下面默认下载：

- LIBERO LeRobot v3：约 35 GB
- RoboCasa GR1 `gr1_unified.*` 24k trajectories
- RoboCasa GR1 仿真环境及 assets

```bash
set -euo pipefail

: "${DATA_ROOT:?请先设置 DATA_ROOT}"

hf download HuggingFaceVLA/libero \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}/libero-lerobot-v3"

hf download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --include "gr1_unified.*/**" \
  --local-dir "${DATA_ROOT}/robocasa-gr1-24k"

if [[ ! -d "${DATA_ROOT}/robocasa-gr1-tabletop-tasks/.git" ]]; then
  git clone \
    https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git \
    "${DATA_ROOT}/robocasa-gr1-tabletop-tasks"
fi

python -m pip install -e "${DATA_ROOT}/robocasa-gr1-tabletop-tasks"
python "${DATA_ROOT}/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py" -y
```

数据链接：

- [LIBERO LeRobot v3](https://huggingface.co/datasets/HuggingFaceVLA/libero)
- [RoboCasa GR1 24k/240k](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
- [RoboCasa GR1 环境](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)

## 4. 可选：下载 240k RoboCasa GR1

完整 Hugging Face 仓库约 1.91 TB。只下载 240k GR1 子集：

```bash
hf download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --include "gr1_arms_waist.*/**" \
  --local-dir "${DATA_ROOT}/robocasa-gr1-240k"
```

## 5. 下载后检查

```bash
du -sh \
  "${DATA_ROOT}/libero-lerobot-v3" \
  "${DATA_ROOT}/robocasa-gr1-24k"

test -f "${DATA_ROOT}/libero-lerobot-v3/meta/info.json"

ROBOCASA_GR1_PATH="${DATA_ROOT}/robocasa-gr1-tabletop-tasks" \
bash benchmarks/robocasa_gr1/run_smoke.sh import
```

NVIDIA GR1 下载目录已经是 LeRobot v2.0（不是 HDF5），但当前 reader
要求 v3 metadata/path contract。用仓库脚本非破坏性转换：

```bash
export ROBOCASA_GR1_PATH="${DATA_ROOT}/robocasa-gr1-tabletop-tasks"
export ROBOCASA_GR1_PYTHON=/path/to/robocasa-gr1/bin/python
export OPENWAM_PYTHON=/path/to/openwam/bin/python
bash scripts/prepare_robocasa_gr1_eef33.sh \
  "${DATA_ROOT}/robocasa-gr1-24k" \
  "${DATA_ROOT}/robocasa-gr1-eef33-v20" \
  "${DATA_ROOT}/robocasa-gr1-eef33-v30"
```

默认使用 hardlink，不复制约 39GB payload。跨文件系统时使用
`--link-mode symlink`。目标根下每个 task bucket 包含：

```text
robocasa-gr1-eef33-v30/<task-bucket>/
├── meta/info.json
├── meta/episodes/
├── data/
└── videos/
```

训练数据契约：

- `eef33_action/state`：`[L xyz3+rot6d6+hand6, R xyz3+rot6d6+hand6, waist3]`
- EEF pose 使用 `robot0_base` 坐标系，避免 world placement 随环境构造漂移
- video: 单个 `observation.images.ego_view`
- prompt: `task_index -> meta/tasks.parquet`
- `annotation.human.coarse_action` 是整数类别，不是 prompt 文本

原始 NVIDIA joint44 不包含 EEF33。转换器会拒绝缺少上述 EEF 列的数据，
不会把 joint 伪装成 xyz+rot6d+gripper。

## 6. 生成 RoboCasa normalization stats

将 `configs/dataloader/robocasa_gr1.yaml` 中的 `dataset_dir` 改为转换后的
目录。配置仅支持 EEF33，并默认映射到 unified80：

```bash
python -m openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation \
  --config configs/dataloader/robocasa_gr1.yaml \
  --output "${DATA_ROOT}/robocasa-gr1-eef33-v30/normalization_stats.npy"
```

随后配置：

```yaml
dataset_dir: /path/to/h200/storage/datasets/robocasa-gr1-eef33-v30
action_mode: eef
unify_action: true
unify_action_map: ["0-8", "10-15", "34-42", "44-49", "68-70"]
normalize_mode: min-max
normalization_stats_path: /path/to/h200/storage/datasets/robocasa-gr1-eef33-v30/normalization_stats.npy
```

stats 脚本会将 rotation6d 的 min/max 固定为 `-1/1`、mean/std 固定为
`0/1`。action command 与 achieved state 分开统计，避免 hand command 和
hand qpos 混用同一分布。

可视化检查：

```bash
python scripts/inspect_robocasa_gr1_dataloader.py \
  --config configs/dataloader/robocasa_gr1.yaml \
  --dataset-dir "${DATA_ROOT}/robocasa-gr1-eef33-v30" \
  --output-dir "${DATA_ROOT}/robocasa-gr1-inspection"
```

## 7. 8×H200 SFT 起始配置

先以 global batch 64、30k optimizer steps 开始：

> 注意：模型训练头为 unified80，部署 normalizer 会 gather 并反归一化为
> EEF33。benchmark client 再通过双臂 IK 转为 29D joint action，并直接透传
> Fourier hand6 与 waist3。

```bash
bash scripts/train.sh \
  dataloader=robocasa_gr1 \
  dataloader.dataset_dir="${DATA_ROOT}/robocasa-gr1-eef33-v30" \
  dataloader.action_mode=eef \
  dataloader.unify_action=true \
  'dataloader.unify_action_map=["0-8","10-15","34-42","44-49","68-70"]' \
  dataloader.normalize_mode=min-max \
  dataloader.normalization_stats_path="${DATA_ROOT}/robocasa-gr1-eef33-v30/normalization_stats.npy" \
  model.architecture.action_dim=80 \
  model.architecture.state_dim=80 \
  training.batch_size=4 \
  training.gradient_accumulation_steps=2 \
  training.num_epochs=null \
  training.max_steps=30000 \
  training.learning_rate=3e-5 \
  training.save_steps=2000 \
  training.mixed_precision=bf16 \
  training.zero_stage=2 \
  training.use_gradient_checkpointing=true \
  training.output_path=/path/to/h200/storage/checkpoints/robocasa-gr1-sft
```

有效 batch：

```text
4（每卡）× 8（H200）× 2（梯度累积）= 64
```

先运行 100 步确认显存：

```bash
# 将上面命令临时改为：
training.max_steps=100 training.save_steps=100
```

若显存充足，可使用：

```text
training.batch_size=8
training.gradient_accumulation_steps=1
```

保持 global batch 64。24k 多任务数据先训练 30k steps，每 2k steps 评测；成功率仍持续上升时再延长至 60k steps。

