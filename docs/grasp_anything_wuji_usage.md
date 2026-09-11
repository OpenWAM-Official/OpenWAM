# Grasp Anything Wuji 训练用法

仓库根目录为 `/gaozt-test1/fyhong/OpenWAM`。

## 转换数据

```bash
cd /gaozt-test1/fyhong/OpenWAM
conda activate openwam

python scripts/prepare_grasp_anything_openwam.py \
  --source data/grasp_anything/grasp_anything_eef_rot6d \
  --destination data/grasp_anything/grasp_anything_eef_rot6d_col
```

默认创建绝对视频软链接，不会修改原始数据。目标目录非空时需显式加 `--overwrite`；需要复制视频时加 `--copy-videos`。

## 验证

```bash
find data/grasp_anything/grasp_anything_eef_rot6d_col/data -name '*.parquet' | wc -l
find data/grasp_anything/grasp_anything_eef_rot6d_col/videos -type l | wc -l
python scripts/train.py dataloader=wuji_real_task --cfg job
```

预期为 75 个 parquet、225 个视频链接，模型 `action_dim` 和 `state_dim` 均为 80。

## 单卡 20-step debug

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Real-Dexterous-Hand-Wuji \
OUTPUT_PATH=outputs/grasp_anything_wuji_debug \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
DATASET_NUM_WORKERS=0 \
OFFLOAD_OPTIMIZER_DEVICE=cpu \
DEBUG=true \
bash scripts/train_grasp_anything_wuji.sh
```

单张约 96 GB H20 使用 ZeRO-2 时需要将 optimizer offload 到 CPU，否则首次分配 Adam 状态会 OOM。确认 loss 有限、三视角视频可读，并检查输出目录含 `config.yaml`、`normalization_stats.npy` 和 safetensors。

## 正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
OUTPUT_PATH=outputs/grasp_anything_wuji \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
NUM_EPOCHS=20 \
LEARNING_RATE=5e-5 \
MIXED_PRECISION=bf16 \
ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
DATASET_NUM_WORKERS=8 \
SAVE_STEPS=1000 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=5 \
bash scripts/train_grasp_anything_wuji.sh
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Real-Dexterous-Hand-Wuji \
OUTPUT_PATH=outputs/grasp_anything_wuji \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
NUM_EPOCHS=20 \
LEARNING_RATE=5e-5 \
MIXED_PRECISION=bf16 \
ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
DATASET_NUM_WORKERS=8 \
SAVE_STEPS=1000 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=5 \
bash scripts/train_grasp_anything_wuji.sh
```

FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Real-Dexterous-Hand-Wuji \

有效 global batch 为 `BATCH_SIZE * GPU 数 * GRADIENT_ACCUMULATION_STEPS`。恢复中断训练时用 `RESUME_CKPT_PATH=<run目录>` 替换 `FINETUNE_CKPT_PATH`，二者不能同时设置。

部署端输入输出是转换后的 raw 58D：`[L EEF9, L hand20, R EEF9, R hand20]`；rot6d 为旋转矩阵前两列，不要再次转换。
