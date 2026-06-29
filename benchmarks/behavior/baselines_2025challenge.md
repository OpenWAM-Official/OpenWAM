# BEHAVIOR-1K 2025 Challenge — Training Configs & Baseline Scores

> Two deliverables: **① 训练步数/batch 配置调研**(供微调测试参照)、**② 已有 baseline 得分**(供效果对比/论文图表)。
> 范围:BEHAVIOR-1K 2025 Challenge(NeurIPS 2025,OmniGibson,机器人 Galaxea R1 Pro,50 长程家务任务)。
> 置信度:✅ 一手源确证 / ⚠️ 单源或示例值 / ❌ 未公布。

---

## ① 训练配置(微调参照)

排名指标见 §②。两位 top 选手均用 **π0.5 (Pi0.5) + 8×H200**;第一名动作空间 **23D**(3 base 速度 + 4 trunk + 7+1 左 + 7+1 右)与本仓库 BEHAVIOR dataloader `action[23]` schema 一致,可直接作微调与 sanity-check 参照。

### 官方 baseline 配方

| 方法 | base | steps | batch | lr / optimizer | 其它 | 硬件 | 置信度 |
|---|---|---|---|---|---|---|---|
| **π0 / openpi** (`pi0_b1k`, 从 `pi0_base` 初始化) | π0 | **50,000** | **64** (CLI flag) | `peak_lr=2.5e-5 → decay 2.5e-6`, warmup 1k, decay 30k, **AdamW**(openpi 默认,配方未显式给出) | action_horizon 50, gemma_2b_lora | 单大显存 GPU(`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`) | ✅ steps/batch/默认 LR |
| **OpenVLA-OFT(+)** | OpenVLA-7B | **100,005** (decay@50k) | **4** | **5e-4** | LoRA rank 32, 3 图, 25-act chunk, L1 reg + FiLM | 未明示 | ✅ |
| ACT / Diffusion Policy(RGB(D)+3D)/ BC-RNN / WB-VIMA | BC | 仅给训练配方,无聚合得分 | — | — | `il_lib` 仓库 | — | ✅ 方法 |

注:官方 index/CFP 写 "OpenVLA",baselines 教程页实为 **OpenVLA-OFT / OFT+** 配方。**无 RL/PPO/SAC 官方 baseline**。π0 配方未写 LR,但 openpi 代码默认即上表值(读 `optimizer.py` 可得)。

### Top-2 竞赛方案

| 方案 | base | steps | batch | lr / optimizer | 数据 | 硬件 | 置信度 |
|---|---|---|---|---|---|---|---|
| **① Robot Learning Collective** (Larchenko 等) | **π0.5**(Gemma 300M action expert ~311M *仅 action expert*;JAX/FSDP) | 按墙钟:多任务 **~15 天** → 每 task-group 微调 **~1 周**,全程 **≈2 epochs**(README 示例 `num_train_steps=200000`,非声明最终值) | 多卡 **2048** / 单卡 **16**(README 示例) | ❌ 未公布 | 10k demos / 50 任务(200 demo/任务)/ 1200+ h;RGB-only 224×224 子集 ~260GB | 训练 **8×H200**(FSDP);推理单张 RTX 4090;评测扩 20×4090 | 硬件/墙钟 ✅;steps/batch ⚠️ 示例;LR ❌ |
| **② Comet** (NVIDIA) | **π0.5**(JAX) | 多任务预训练 **50k**;单任务 SFT **15k–20k** | 论文 per-device **64**;⚠️ 释出 config 实为 8×32(32/卡,256 总) | 预训练 **2.5e-5**,SFT/RFT **2.5e-6**,**cosine**;AdamW | 官方 10k + 自采 ~3.6K(规划器+离线RL)+ RFT 3 轮(~2.5K 选样) | **8×H200** | ✅(论文);batch ⚠️ 论文 vs config |

**①方案要点**:两阶段(50 任务多任务预训练 → 拆 4 个 task-group 专用 ckpt,最终提交 = 4 ckpt,按 task ID 自动切换);horizon 30,3 路相机(头+双腕 224×224);correlated noise flow matching、learnable mixed-layer attention、可学习任务嵌入(vision-action,非语言 VLA);预算 ~$13k(Nebius 赞助 $10k)。

---

## ② Baseline 得分(效果对比 / 论文图表)

### 竞赛 leaderboard(官方,排名指标 = held-out test **Q-score**)

口径:50 任务 ×200 训练 demo;评测 = 10 public-validation + 10 held-out(private)实例/任务,仅 top-5 跑 held-out。**Q-score** = 已满足 BDDL 目标谓词数 / 总目标谓词数,50 任务平均(给部分分)。**Full Success** = 二元全成功,**不用于排名**。赛道:**Standard**(RGB-D+seg+proprio,禁全局位姿)/ **Privileged**(可查仿真器特权信息)。官方页标注 "Provisional"。

| 排名 | 队伍 | 所属 | 赛道 | Full Success (pub / test) | **Q-score (pub / test)** | 置信度 |
|---|---|---|---|---|---|---|
| 1 | **Robot Learning Collective** (Larchenko, Zarin, Karnatak) | Independent | Standard | 0.1120 / 0.1240 | 0.2605 / **0.2599** | ✅ |
| 2 | **Comet** | NVIDIA Research | Standard | 0.1440 / 0.1140 | 0.1830 / **0.2514** | ✅ |
| 3 | SimpleAI Robot | Beijing Simple AI | Standard | 0.1400 / 0.1080 | 0.1943 / **0.1591** | ✅ |
| 4 | The North Star | Huawei CRI EAI | Standard | 0.1280 / 0.0760 | 0.1702 / **0.1204** | ✅ |
| 5 | Embodied Intelligence | Independent | **Privileged** | 0.0620 / 0.0520 | 0.1110 / **0.0947** | ✅ |

规模:18 队 / 4 国(美·中·加·韩);每赛道各取 top-3;现金奖 $1,000 / $500 / $300。
排名按 **held-out test Q-score**(①0.2599 > ②0.2514);注意 ② 的 pub-val full-success 0.1440 高于 ① 0.1120,但非排名指标。
⚠️ Comet GitHub 提到的 **0.345** 是赛后(post-challenge)public-validation 双模型分,**非** held-out test(0.2514),论文未收录,勿引用。

### 官方 baseline 聚合得分

❌ **官方未公布 π0 / OpenVLA-OFT / ACT / DP / BC-RNN / WB-VIMA 的 50 任务聚合得分**——仅提供个别单任务 checkpoint(π0:turning_on_radio + picking_up_trash,各 50k step;WB-VIMA:仅 turning_on_radio)。
若论文需 "baseline floor" 弱下限,可引榜单第 13 名参赛队 "ACT"(Xiamen,Q≈0.0037),**须注明是参赛实现,非官方 baseline**。

### 可引的 RL 对照(2024 原始论文,口径不同,不可与 2025 直接比)

arXiv:2403.09227 Table 2,**RL-Prim.Hist.**(PPO+原语+历史):StoreDecoration / CollectTrash / CleanTable = **0.55 / 0.63 / 0.88**(✅)。仅 3 个活动、用 assistive/sticky 抓取原语,与 2025(50 任务、R1 Pro、纯 IL、无抓取辅助)**不可比**,引用须加注。

---

## Sources

**官方** — leaderboard https://behavior.stanford.edu/challenge/leaderboard.html · evaluation https://behavior.stanford.edu/challenge/evaluation.html · baselines https://behavior.stanford.edu/challenge/baselines.html · index https://behavior.stanford.edu/challenge/index.html · dataset https://behavior.stanford.edu/challenge/dataset.html · HF https://huggingface.co/datasets/behavior-1k/2025-challenge-demos
**① RLC** — arXiv 2512.06951 · GitHub https://github.com/IliaLarchenko/behavior-1k-solution · HF https://huggingface.co/IliaLarchenko/behavior_submission
**② Comet** — arXiv 2512.10071 · GitHub https://github.com/mli0603/openpi-comet
**官方 baseline 仓库 / 原始论文** — https://github.com/StanfordVL/b1k-baselines (openpi fork: github.com/wensi-ai/openpi `behavior` 分支;`il_lib`: github.com/wensi-ai/il_lib) · https://github.com/StanfordVL/BEHAVIOR-1K · 2024 RL baselines: arXiv 2403.09227
