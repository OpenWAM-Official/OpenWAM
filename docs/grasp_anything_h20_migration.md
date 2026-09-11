# Grasp Anything / Wuji 58D EEF 适配迁移到 H20 服务器

本文档用于让另一台服务器上的 Codex 从**原版 OpenWAM 仓库**重新完成本项目的
数据与训练适配。目标服务器已经装好 `openwam` conda 环境并下载好模型，因此不需要
重新安装环境或下载 checkpoint。

本文档中的绝对路径以目标服务器为准：

```text
仓库根目录:
/gaozt-test1/fyhong/OpenWAM

原始 LeRobot v2 EEF 数据（rot6d 前两行）:
/gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d

需要生成的 OpenWAM 数据（rot6d 前两列）:
/gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col
```

本迁移规格记录自 `OpenWAM-Official/OpenWAM` commit：

```text
75ed9a7c29e76480c2eb64cb10ae06fb03996e14
```

目标仓库不必强制回退到这个 commit，但 Codex 应先记录目标 `git rev-parse HEAD`，再检查
本文涉及的接口是否发生变化。目标仓库有用户改动时不得 reset 或覆盖。

## 1. 给目标服务器 Codex 的任务边界

在目标仓库中完成以下工作，并在所有验收通过前不要启动正式训练：

1. 检查目标仓库当前版本的 dataloader registry、`LeRobotV3Reader`、统一 80D action
   映射和部署 normalizer 接口，保留目标仓库已有改动。
2. 新增 Wuji 58D rot6d 行/列转换工具和离线数据转换脚本。
3. 从原始 `grasp_anything_eef_rot6d` 生成独立的
   `grasp_anything_eef_rot6d_col`；不得原地修改原数据。
4. 新增 `wuji_real_task` reader 和 Hydra dataloader 配置，并注册 reader。
5. 新增便于调整 batch size、学习率、保存频率等参数的训练 wrapper。
6. 验证数据数量、tensor shape、58D 到 80D 的映射、rot6d 正交性和 Hydra 配置。
7. 先执行单卡 20-step debug，再启动正式训练。

如果目标仓库接口与本文档不同，应按目标仓库现有抽象适配，但下文定义的**数据语义、
维度映射、rot6d 数学和验收结果不能改变**。

可直接给目标服务器 Codex 的入口指令为：

```text
请完整阅读 docs/grasp_anything_h20_migration.md，并以它作为实施和验收规格。
先检查当前 OpenWAM 版本及已有修改，然后完成第 1 至 5 节的适配与验证；
不要修改原始数据，不要在验证通过前启动正式训练，也不要覆盖我已有的仓库改动。
模型和 conda openwam 环境已经准备好。
```

## 2. 数据契约

### 2.1 原始 58D 布局

原始 GR00T/Astribot 数据的 `action` 和 `observation.state` 都是 58D：

```text
[0:9]   左臂 EEF:  xyz(3) + rot6d-row(6)
[9:18]  右臂 EEF:  xyz(3) + rot6d-row(6)
[18:38] 左手绝对关节(20)
[38:58] 右手绝对关节(20)
```

EEF 与手指均是绝对量。这里不应计算 delta，也不应增加机器人基座/世界坐标变换。

### 2.2 转换后的 58D 布局

目标 bucket 中的 `action` 和 `observation.state` 改为：

```text
[0:9]   左臂 EEF:  xyz(3) + rot6d-column(6)
[9:29]  左手绝对关节(20)
[29:38] 右臂 EEF:  xyz(3) + rot6d-column(6)
[38:58] 右手绝对关节(20)
```

也就是：

```text
[L EEF9, R EEF9, L hand20, R hand20]
                         -> [L EEF9, L hand20, R EEF9, R hand20]
```

转换后的 rot6d 维度为 `[3:9]` 和 `[32:38]`。

### 2.3 rot6d 行转列算法

原始 6D 是旋转矩阵的前两行：

```text
[R00, R01, R02, R10, R11, R12]
```

OpenWAM 需要前两列：

```text
[R00, R10, R20, R01, R11, R21]
```

不能只重新排列已有六个数，因为 `R20` 和 `R21` 不在原始 6D 中。正确做法是：

```python
r0 = normalize(x[..., 0:3])
r1 = x[..., 3:6] - dot(r0, x[..., 3:6]) * r0
r1 = normalize(r1)
r2 = cross(r0, r1)
rot6d_col = concat([
    r0[..., 0:1], r1[..., 0:1], r2[..., 0:1],
    r0[..., 1:2], r1[..., 1:2], r2[..., 1:2],
])
```

使用 `float32`，归一化分母下限为 `1e-8`。该 Gram-Schmidt 投影用于消除采集数据中的
微小非正交误差。转换不会改变坐标系，只改变同一个旋转矩阵的 6D 存储约定。

### 2.4 OpenWAM canonical 80D 映射

模型 action/state head 保持 OpenWAM checkpoint 的 80D，不要改为 58D。转换后的 58D
依次散射到以下槽位：

| 原始目标块 | 58D source | 80D destination |
| --- | ---: | ---: |
| 左 EEF | `0:9` | `0:9` |
| 左手 | `9:29` | `10:30` |
| 右 EEF | `29:38` | `34:43` |
| 右手 | `38:58` | `44:64` |

Hydra 配置必须是：

```yaml
unify_action: true
unify_action_map:
  - "0-8"
  - "10-29"
  - "34-42"
  - "44-63"
```

这些区间在 YAML 中是 inclusive，合计 `9 + 20 + 9 + 20 = 58` 维。不要采用默认的
`0:58` identity mapping，否则右臂和手指会落入错误的 canonical 槽位。

## 3. 需要新增或修改的仓库文件

当前适配涉及以下文件：

```text
新增 openwam/dataloader/utils/rot6d.py
新增 scripts/prepare_grasp_anything_openwam.py
新增 openwam/dataloader/wuji_real_task.py
新增 configs/dataloader/wuji_real_task.yaml
新增 scripts/train_grasp_anything_wuji.sh
修改 openwam/dataloader/registry.py
修改 openwam/dataloader/__init__.py
```

如果可以从旧服务器复制代码，这是最稳妥的方式。应只复制上述文件，并对 registry 和
`__init__.py` 用 diff/patch 合并，不能覆盖目标仓库中可能已有的其他修改。复制后仍必须把
配置里的 `dataset_dir` 更新为目标路径，并执行本文档的全部验收。

如果只将本文档交给目标服务器 Codex，则按下面的接口规格重新实现。

### 3.1 `openwam/dataloader/utils/rot6d.py`

至少实现以下函数：

```python
row_rot6d_to_col_rot6d(values: np.ndarray) -> np.ndarray
convert_wuji_58(values: np.ndarray, *, row_rot6d: bool = True) -> np.ndarray
wuji_58_rot6d_to_matrix(values: np.ndarray) -> np.ndarray
```

要求：

- 输入最后一维不符合 6 或 58 时立即报错。
- `convert_wuji_58` 同时完成两只 EEF 的 rot6d 转换和 58D block 重排。
- 输出统一为 `np.float32`。
- `wuji_58_rot6d_to_matrix` 从转换后布局的 `[3:9]`、`[32:38]` 恢复两个 `3x3`
  旋转矩阵，供验收使用。

### 3.2 `scripts/prepare_grasp_anything_openwam.py`

转换脚本应使用 `pyarrow` 读写 parquet，不要逐字段进行文本处理。要求：

- 参数：`--source`、`--destination`、`--copy-videos`、`--overwrite`。
- 对每个 episode 的 `action` 和 `observation.state` 调用 `convert_wuji_58`。
- 原始 parquet 的其他字段原样保留。
- 输出每个 episode 一个 parquet，使用 zstd 压缩。
- 默认将三个视角视频以绝对软链接放入目标 bucket，避免重复占用磁盘。
- `--copy-videos` 改为复制视频，适合目标 bucket 之后还需要独立迁移的情况。
- 复制 `tasks.jsonl`，重写 `episodes.jsonl` 并标记
  `action_space=eef_absolute_hand_absolute_rot6d_column`。
- 生成 `meta/info.json`、`meta/modality.json`、`meta/stats.json`、
  `meta/normalization_stats.npy` 和 `meta/adapter_manifest.json`。

目标 `info.json` 的关键内容：

```json
{
  "codebase_version": "openwam-grasp-anything-v2-adapter-1",
  "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
  "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4",
  "robot_type": "WUJI_ASTRIBOT_EEF_ABSOLUTE_HAND_ABSOLUTE_ROT6D_COLUMN",
  "splits": {"train": "0:75"}
}
```

`fps`、episode/frame 数和 `features` 应从源数据推导，不要无条件写死 75。已知当前这份
数据应得到 75 episodes、63725 frames、30 FPS。

归一化统计用转换后的所有 action 和 state 行合并计算：`mean/std/min/max/q01/q99`。
rot6d 是几何表示，不应作为 12 个独立标量进行缩放，因此以下目标 58D 维度必须固定为：

```text
rot6d dims = 3,4,5,6,7,8,32,33,34,35,36,37
mean=0, std=1, min=-1, max=1, q01=-1, q99=1
```

### 3.3 `openwam/dataloader/wuji_real_task.py`

优先继承目标仓库现有的 `LeRobotV3Reader`，实现单 bucket reader：

```python
DATASET_NAME = "WujiRealTask"
ACTION_DIM = 58
NEEDED_COLS = ("action", "observation.state", "task_index")
DEPLOY_ACTION_MODE = "eef"
```

相机字段：

```text
observation.images.head_view
observation.images.left_wrist_view
observation.images.right_wrist_view
```

reader 需要：

- 从 `meta/episodes.jsonl` 构造 episode index。
- 从 `meta/tasks.jsonl` 读取 `task_index -> task` prompt。
- 从 `meta/normalization_stats.npy` 读取 58D `eef` 统计。
- `normalize_mode=quantile` 时对 action/state 归一化，但保持 rot6d 维度不变。
- action 读取整段 window；proprio 只读取 window 第一帧。
- 若配置 `rot6d_convention=row`，支持在线行转列作为兼容路径；正式配置用 `column`。
- 当 `unify_action=true` 时使用本文档的 canonical map，将 action/proprio scatter 为 80D。
- 检查 `robot_type` 和 `action`/`observation.state` shape，格式不符时立即失败。
- 将 `normalization_stats_path` 暴露给 trainer，使新 checkpoint 带有部署所需的统计文件。

如果目标仓库的部署端已经支持 `_UnifyAwareNormalizer`，部署时应执行：

```text
机器人 raw 58D state -> normalize(raw) -> scatter 到 80D -> 模型
模型 80D action -> gather 回 raw 58D -> unnormalize(raw) -> 机器人
```

若原版部署端不支持上述 gather/scatter，需要一并移植或实现等价逻辑；否则训练能跑通，
但真机部署时会得到错误维度。

在 commit `75ed9a7` 中，该逻辑位于 `openwam/deploy/model_loader.py`。目标 Codex 应确认：

- `_build_normalizer` 在 `dataloader.unify_action=true` 时解析 `unify_action_map`。
- 输入 proprio 执行 raw normalize 后 scatter 到 80D。
- 模型输出执行 80D gather 后再按 raw 58D stats unnormalize。
- 缺少 `normalization_stats.npy` 或缺少 `eef` stats key 时明确报错，不允许静默关闭归一化。

### 3.4 registry 和导出

在 `openwam/dataloader/registry.py` 的 builtin 注册函数中导入并注册：

```python
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
register_dataset("wuji_real_task")(WujiRealTaskDataset)
```

在 `openwam/dataloader/__init__.py` 中导入，并加入 `__all__`。修改前先查看目标文件，
只做增量编辑。

### 3.5 Hydra 配置

新增 `configs/dataloader/wuji_real_task.yaml`：

```yaml
type: wuji_real_task
dataset_dir: /gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col

action_mode: eef
rot6d_convention: column
num_frames: 33
video_stride: 4
window_stride: 1
height: 384
width: 320
multiview: true
target_camera: observation.images.head_view
camera_layout:
  - observation.images.head_view
  - observation.images.left_wrist_view
  - observation.images.right_wrist_view
normalize_mode: quantile
enable_action_supervision: true
unify_action: true
unify_action_map:
  - "0-8"
  - "10-29"
  - "34-42"
  - "44-63"
color_jitter: false
```

模型配置应继续使用 `action_dim: 80` 和 `state_dim: 80`。

### 3.6 训练 wrapper

新增 `scripts/train_grasp_anything_wuji.sh`，内部调用：

```bash
bash scripts/train.sh dataloader=wuji_real_task <Hydra overrides...>
```

至少暴露以下环境变量：

```text
FINETUNE_CKPT_PATH, RESUME_CKPT_PATH, OUTPUT_PATH
NPROC_PER_NODE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
NUM_EPOCHS, MAX_STEPS, LEARNING_RATE, WEIGHT_DECAY
ACTION_LR, VIDEO_LR, WARMUP_RATIO, LR_MIN_RATIO
MIXED_PRECISION, ZERO_STAGE, USE_GRADIENT_CHECKPOINTING
USE_GRADIENT_CHECKPOINTING_OFFLOAD, INITIALIZE_MODEL_ON_CPU
OFFLOAD_OPTIMIZER_DEVICE, DATASET_NUM_WORKERS
SAVE_STEPS, SAVE_FULL_STATES_FOR_RESUME, KEEP_LAST_K_CKPTS, DEBUG
```

`FINETUNE_CKPT_PATH` 和 `RESUME_CKPT_PATH` 必须互斥。wrapper 最后保留 `"$@"`，允许
追加任意 Hydra override。

## 4. 在 H20 服务器转换数据

目标服务器把原数据放好后，先做最小环境与 GPU 检查：

```bash
cd /gaozt-test1/fyhong/OpenWAM
conda activate openwam

python - <<'PY'
import torch, hydra, omegaconf, numpy, pandas, pyarrow
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory // 2**30, "GiB")
PY

nvidia-smi
```

确认 `pyarrow` 等导入成功、CUDA 可用且显示的是预期 H20。然后执行转换：

```bash
cd /gaozt-test1/fyhong/OpenWAM
conda activate openwam

python scripts/prepare_grasp_anything_openwam.py \
  --source /gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d \
  --destination /gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col
```

默认目标视频是指向同机原数据的软链接，因此不要在训练期间移动或删除
`grasp_anything_eef_rot6d`。若目标数据必须自包含，首次转换时增加 `--copy-videos`。

目标目录非空时脚本应拒绝覆盖。只有确认它是失败或过期的转换结果后才使用：

```bash
python scripts/prepare_grasp_anything_openwam.py \
  --source /gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d \
  --destination /gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col \
  --overwrite
```

## 5. 数据与代码验收

### 5.1 目录和数量

对当前这份数据应看到：

```bash
find data/grasp_anything/grasp_anything_eef_rot6d_col/data -name '*.parquet' | wc -l
find data/grasp_anything/grasp_anything_eef_rot6d_col/videos -type l | wc -l
```

期望分别为 `75` 和 `225`。如果使用 `--copy-videos`，第二条应改用 `-type f`。

检查软链接无失效项：

```bash
find -L data/grasp_anything/grasp_anything_eef_rot6d_col/videos -type l -print
```

期望无输出。

### 5.2 rot6d 和转换一致性

至少读取一个源/目标 episode，验证：

- `convert_wuji_58(source_action)` 与目标 parquet action 最大绝对误差接近 0。
- 恢复的旋转矩阵满足 `R.T @ R = I`、`det(R) = 1`。
- 当前数据实测误差量级为 `1e-7`；不同批次或实现不要求 bitwise 相等，但正交误差和
  source/target 恢复误差都应在 `1e-5` 以下。

### 5.3 reader smoke test

```bash
python - <<'PY'
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset

cfg = dict(
    dataset_dir="/gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col",
    num_frames=33,
    video_stride=4,
    window_stride=1,
    height=384,
    width=320,
    multiview=True,
    target_camera="observation.images.head_view",
    camera_layout=[
        "observation.images.head_view",
        "observation.images.left_wrist_view",
        "observation.images.right_wrist_view",
    ],
    normalize_mode="quantile",
    enable_action_supervision=True,
    rot6d_convention="column",
    unify_action=True,
    unify_action_map=["0-8", "10-29", "34-42", "44-63"],
)
ds = WujiRealTaskDataset.from_config(cfg)
sample = ds[0]
print("length:", len(ds))
print("action:", sample["action"].shape, "valid:", sample["action_mask"].sum().item())
print("proprio:", sample["proprio"].shape, "valid:", sample["proprio_mask"].sum().item())
print("video frames:", len(sample["video"]))
PY
```

当前数据的期望结果：

```text
length: 63725
action: torch.Size([32, 80]) valid: 1856
proprio: torch.Size([1, 80]) valid: 58
video frames: 9
```

再验证 scatter/gather 后 58D 的最大误差为 0，并确认 mask 只激活 canonical map 的
58 个维度。

### 5.4 Hydra 和 checkpoint

```bash
python scripts/train.py dataloader=wuji_real_task --cfg job
```

该命令只打印配置，不启动训练。确认输出包含：

```text
dataloader.type: wuji_real_task
dataloader.dataset_dir: .../grasp_anything_eef_rot6d_col
model.architecture.action_dim: 80
model.architecture.state_dim: 80
```

分别检查 foundation 和 Wuji checkpoint 目录：

```bash
find /path/to/checkpoint -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

必须至少有 `config.yaml` 和完整的 `checkpoint_step_*.safetensors`。存在 `.aria2` 临时文件
通常意味着下载尚未完成。Wuji checkpoint 还通常带 `normalization_stats.npy`。

不要为了训练而直接改写下载 checkpoint 的 `config.yaml`。`finetune_ckpt_path` 应从
checkpoint 重建模型，但本次 run 的 `dataloader=wuji_real_task` 和新数据统计应写入新的
输出目录。只有目标 OpenWAM 版本错误地让 checkpoint dataloader 覆盖当前 dataloader 时，
才应修复加载/merge 逻辑，而不是静默训练错误布局。

## 6. 选择 GPU、保存频率和训练命令

`CUDA_VISIBLE_DEVICES` 决定使用物理哪几张卡，`NPROC_PER_NODE` 必须等于可见卡数量。
例如选择物理 GPU 2、3、6、7：

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 \
NPROC_PER_NODE=4 \
FINETUNE_CKPT_PATH=/path/to/OpenWAM-Alpha-Real-Dexterous-Hand-Wuji \
OUTPUT_PATH=outputs/grasp_anything_wuji \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_EPOCHS=20 \
MAX_STEPS=null \
LEARNING_RATE=5e-5 \
MIXED_PRECISION=bf16 \
ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
DATASET_NUM_WORKERS=8 \
SAVE_STEPS=500 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=2 \
  bash scripts/train_grasp_anything_wuji.sh
```

上面的命令**每 500 个 global micro-step 保存一次模型**，保留最近 2 个中间 checkpoint，
并同时保存可 resume 的 optimizer/scheduler/RNG 状态。这里 trainer 的 `global_step` 每读取
一个 micro-batch 增加 1；当梯度累积为 4 时，500 micro-step 约等于 125 optimizer step。
建议 `SAVE_STEPS` 取梯度累积步数的整数倍。

该命令的有效 global batch 为：

```text
BATCH_SIZE * GPU 数 * GRADIENT_ACCUMULATION_STEPS = 1 * 4 * 4 = 16
```

使用 foundation checkpoint 时仅替换路径和输出目录：

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 \
NPROC_PER_NODE=4 \
FINETUNE_CKPT_PATH=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model \
OUTPUT_PATH=outputs/grasp_anything_foundation \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_EPOCHS=20 \
LEARNING_RATE=5e-5 \
SAVE_STEPS=500 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=2 \
  bash scripts/train_grasp_anything_wuji.sh
```

H20 上也应先从保守的 `BATCH_SIZE=1` 开始，根据实际峰值显存逐步增加。增加 GPU 数不会
减少单卡模型/激活占用；单卡 OOM 时优先降低 `BATCH_SIZE`，保持 gradient checkpointing，
必要时启用 optimizer CPU offload。

### 6.1 首次 20-step debug

```bash
CUDA_VISIBLE_DEVICES=2 \
NPROC_PER_NODE=1 \
FINETUNE_CKPT_PATH=/path/to/checkpoint \
OUTPUT_PATH=outputs/grasp_anything_debug \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
DATASET_NUM_WORKERS=0 \
DEBUG=true \
  bash scripts/train_grasp_anything_wuji.sh
```

当前 trainer 在 `DEBUG=true` 时强制运行 20 micro-step，在 step 10 和最终 step 20 保存，
并使用常数学习率。必须检查 loss 为有限值、三视角视频成功读取且输出目录含 config、统计
与 safetensors，再开始正式训练。

### 6.2 resume

首次训练必须设置 `SAVE_FULL_STATES_FOR_RESUME=true` 才能严格恢复。中断后传入实际 run
目录，而不是某个 safetensors 文件：

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 \
NPROC_PER_NODE=4 \
RESUME_CKPT_PATH=outputs/grasp_anything_wuji/<timestamped_run_dir> \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_EPOCHS=20 \
SAVE_STEPS=500 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=2 \
  bash scripts/train_grasp_anything_wuji.sh
```

`FINETUNE_CKPT_PATH` 是只加载权重并从 step 0 创建新 run；`RESUME_CKPT_PATH` 是恢复完整
状态并复用原 run，二者不能同时设置。

## 7. 真机部署前的最终契约

训练产生的实际 run 目录可用于：

```bash
bash scripts/deploy.sh outputs/grasp_anything_wuji/<timestamped_run_dir>
```

真机客户端必须满足：

- 输入和输出是转换后布局的 raw 58D：`[L EEF9, L hand20, R EEF9, R hand20]`。
- 两个 EEF rot6d 都是旋转矩阵前两列；不能在客户端再次做行转列。
- EEF 位置单位、手指关节单位、左右臂定义和数据采集时完全一致。
- 若机器人控制接口仍采用原 GR00T 顺序
  `[L EEF9, R EEF9, L hand20, R hand20]`，客户端边界必须显式重排，不能把模型的
  58D 输出直接发送给该接口。
- 本数据与同一 Astribot/Wuji 真机共用坐标系时，不增加 base/world 坐标变换。只有部署
  机器人或控制器坐标定义实际不同，才增加经过标定的坐标变换。
- 部署 checkpoint 目录必须携带本次数据生成的 `normalization_stats.npy`，并使用与训练相同
  的 `unify_action_map`。部署服务最终应输出已 gather、已反归一化的 raw 58D action。

## 8. 完成标准

目标服务器 Codex 完成后应报告：

1. 实际修改文件清单和与目标原版 OpenWAM 的接口差异。
2. 数据转换后的 episode、frame、parquet 和视频数量。
3. rot6d 正交误差与 source/target 恢复误差。
4. reader 的 action/proprio/video shape 和两个 mask 的有效元素数。
5. Hydra composition 结果与两个 checkpoint 的完整性检查。
6. 单卡 20-step debug 的 loss、显存峰值和 checkpoint 输出目录。
7. 最终建议的 `CUDA_VISIBLE_DEVICES`、batch size、梯度累积和有效 global batch。

在第 1 至 5 项没有通过前，不应消耗 H20 资源启动正式训练。
