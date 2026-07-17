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
├── robocasa-gr1-v30/          # OpenWAM v3 re-index
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
python scripts/convert_robocasa_gr1_v20_to_v30.py \
  --input "${DATA_ROOT}/robocasa-gr1-24k" \
  --output "${DATA_ROOT}/robocasa-gr1-v30"
```

默认使用 hardlink，不复制约 39GB payload。跨文件系统时使用
`--link-mode symlink`。目标根下每个 task bucket 包含：

```text
robocasa-gr1-v30/<task-bucket>/
├── meta/info.json
├── meta/episodes/
├── data/
└── videos/
```

真实数据契约：

- `observation.state`: 44D native joint/body vector
- `action`: 44D native joint/body vector
- video: 单个 `observation.images.ego_view`
- prompt: `task_index -> meta/tasks.parquet`
- `annotation.human.coarse_action` 是整数类别，不是 prompt 文本

数据中没有 EEF20，因此不能把它直接解释成
`xyz + rotation6d + gripper`，也不能直接 scatter 到 EEF80。

## 6. 生成 RoboCasa normalization stats

将 `configs/dataloader/robocasa_gr1.yaml` 中的 `dataset_dir` 改为转换后的
目录。默认配置为 native `joint` / 44D，然后执行：

```bash
python -m openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation \
  --config configs/dataloader/robocasa_gr1.yaml \
  --output "${DATA_ROOT}/robocasa-gr1-v30/normalization_stats.npy"
```

随后配置：

```yaml
dataset_dir: /path/to/h200/storage/datasets/robocasa-gr1-v30
action_mode: joint
action_dim: 44
unify_action: false
unify_action_map: null
normalize_mode: quantile
normalization_stats_path: /path/to/h200/storage/datasets/robocasa-gr1-v30/normalization_stats.npy
```

只有另行生成了真实 EEF 列
`eef_sim_pose_{action,state}[12] + gripper_open_scale_{action,state}[2]`
时，才使用 `configs/dataloader/robocasa_gr1_unify.yaml`。该配置显式映射
EEF20 到 unified `0-9,34-43`；stats 脚本会保持 rotation6d 原样，只归一化
xyz/gripper。

可视化检查：

```bash
python scripts/inspect_robocasa_gr1_dataloader.py \
  --config configs/dataloader/robocasa_gr1.yaml \
  --dataset-dir "${DATA_ROOT}/robocasa-gr1-v30" \
  --output-dir "${DATA_ROOT}/robocasa-gr1-inspection"
```

## 7. 8×H200 SFT 起始配置

先以 global batch 64、30k optimizer steps 开始：

> 注意：24k 数据是 44D joint/body，而当前 tabletop gym wrapper 暴露的
> arms+waist action space 是 29D。下面命令能训练原生 44D checkpoint，
> 但不能在没有明确 44D→29D 部件选择/控制契约时直接送入该 env。

```bash
bash scripts/train.sh \
  dataloader=robocasa_gr1 \
  dataloader.dataset_dir="${DATA_ROOT}/robocasa-gr1-v30" \
  dataloader.action_mode=joint \
  dataloader.action_dim=44 \
  dataloader.unify_action=false \
  dataloader.normalize_mode=quantile \
  dataloader.normalization_stats_path="${DATA_ROOT}/robocasa-gr1-v30/normalization_stats.npy" \
  model.architecture.action_dim=44 \
  model.architecture.state_dim=44 \
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

