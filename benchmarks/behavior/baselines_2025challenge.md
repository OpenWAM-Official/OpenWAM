# BEHAVIOR-1K 2025 Challenge — 有实测分数的策略 · 训练配置与表现

> 本文只收录**在 BEHAVIOR-1K 上有实测分数**的策略（world-model / 通用 VLA peers 不在此文）。
> 分两类：
>
> - **表 A = Challenge** —— 官方 2025 leaderboard 名次，排名口径 = **held-out / private test Q-score**。
> - **表 B = 自测** —— 策略自报的 BEHAVIOR 分数，**非**官方 held-out 排名。
>
> **Q-score** = 已满足 BDDL 目标谓词数 / 总目标谓词数，50 任务平均（部分给分）。
> **口径不可混用**：held-out(private) ≠ public-validation ≠ 自测。
> 置信度：✅ 一手源确证 · ⚠️ 转载或单源 · ❌ 未披露。

---

## 表 A — Challenge（官方 2025 leaderboard）

| 名次 | 方法 | 所属 | 赛道 | 可引用 | **held-out Q** | pub-val Q | Full-Success (priv / pub) |
|:--:|---|---|---|---|:--:|:--:|:--:|
| 1 | **RLC / Robot Learning Collective** | 独立 | Standard | ✅ arXiv:2512.06951 | **0.2599** | 0.2605 | 0.1240 / 0.1120 |
| 2 | **Comet** | NVIDIA | Standard | ✅ arXiv:2512.10071 | **0.2514** | 0.1830 | 0.1140 / 0.1440 |
| 3 | **SimpleAI Robot** | Beijing Simple AI | Standard | ❌ 无 | **0.1591** | 0.1943 | 0.1080 / 0.1400 |
| 4 | **The North Star** | Huawei CRI EAI | Standard | ❌ 无 | **0.1204** | — | 0.076 / — |
| 5 | **Embodied Intelligence** | 独立 | ⚠️ Privileged | ❌ 无 | **0.0947** | 0.1110 | 0.0520 / 0.0620 |

> - 只有 **1、2 名**有可引用 artifact；3 / 4 / 5 名仅存于官方 leaderboard 与转载表（Comet Table 1 / RLC 博客），**训练配置全部未披露**。
> - **#5 为 Privileged track**（可查特权仿真信息），与 1–4 名的 Standard track 不同赛道，谨慎并列。

### 训练配置详情（仅 1、2 名有报告）

#### 1 · RLC / Robot Learning Collective — Larchenko, Zarin, Karnatak（独立）

- **base**：π0.5（SigLIP-So400m/14 冻结 + PaliGemma VLM + Gemma-300M flow-matching expert；语言头替换为 50 个可训练 2048-D task embedding）
- **硬件**：训练 **8×H200**（FSDP）；推理单张 4090；评测 20×4090
- **batch**：⚠️ **无 declared-final** —— README 的 16（单卡）/ 2048（FSDP8）**明标"示例"**，dataclass 默认 32，获胜 config 未覆盖
- **iters / epochs**：config target = **200k steps**，但实际提交是**时间受限**：连续训练约 1 个月、**≈2 epoch、未收敛**
- **两阶段**：50 任务多任务预训练 ~15 天 → 拆 4 个 task-group 各微调 ~1 周（最终提交 = **4 个 ckpt**，按 task 自动切换）
- **优化**：AdamW + CosineDecay（warmup 1000，**peak 1e-4 → 1e-5，decay@20k**）；EMA 0.99；flow-matching t~Beta(1.5,1)，num_flow_samples=15；action horizon 30
- **表现**：**held-out Q 0.2599（第 1）**，pub-val 0.2605（公私几乎无差）；Full-Success（非排名指标）priv 0.1240 / pub 0.1120

#### 2 · Comet — Team Comet（NVIDIA）

- **base**：π0.5（transformer flow-matching action head）
- **硬件**：SFT / RFT 明确 **8×H200**；⚠️ **预训练卡数未披露**（旧传 "gpu40" 经复核为**捏造**，勿引用）
- **batch**：paper 声明 per-device **64**；公开 config 一律 hard-code **8×32 = 256**（全局，可复现值）
- **iters / epochs**（steps，非 epoch）：预训练 **50k** → 单任务 SFT **15–20k** → RFT **20k**；RFT 外循环 **3 轮**
- **优化**：CosineDecay；预训练 peak **2.5e-5**，SFT / RFT peak **2.5e-6**；AdamW；action chunk 32；绝对关节动作；30 Hz；头相机 720 / 腕相机 480 分辨率为关键
- **表现（三口径务必分开）**：
  - **held-out TEST Q 0.2514（第 2）**，完成 22/50，Full-Success 0.1140
  - 赛中 public-val Q 0.1830，Full-Success 0.1440
  - 赛后 public-val Q **0.3453**（仅 2 个 ckpt，**非**挑战 / test 分数，**勿**与他队 test 并列）

---

## 表 B — 自测（策略自报，非官方 held-out 排名）

| 方法 | 可引用 | base | **分数（口径）** | 备注 |
|---|---|---|:--:|---|
| **Galaxea G0.5** | ✅ tech-report URL（无 arXiv；G0 = arXiv:2509.00576） | Qwen3.5-2B 单一自回归 decoder | **0.3136**（4 ep，自测） | 单 generalist ckpt，2 次平均 |
| ↳ G0.5 协议下 re-report | — | — | G0.5(1ep) 0.2904 · pi0.5(4ep) 0.2626 · RLC 0.2605 · Comet 0.1830 | ⚠️ 均 public-val 级，**非** held-out |
| **LEGACY RL**（VMC / Prim. / Prim.Hist.）⚠️ 不可比 | ✅ arXiv:2403.09227 | SAC / PPO + primitive | Prim.Hist. Q **0.59 / 0.68 / 0.88** | 旧 3 任务协议 |

### 训练配置详情

#### Galaxea G0.5（自测口径）

- **引用**：tech report `opengalaxea.github.io/G05/`（**无 arXiv / DOI**；前身 G0 = arXiv:2509.00576，建议双引）
- **base**：**Qwen3.5-2B 单一统一自回归 decoder**（VLM-as-actor；27-D 统一动作空间 + 可学 cross-embodiment tokenizer）。⚠️ **不是**双系统 + action expert —— 那是前身 G0
- **iters / epochs**：BEHAVIOR 后训练 **1 epoch 与 4 epochs** 两档，co-train 全部 10000 episodes / 50 任务；预训练 ~120k steps
- **优化**：预训练 AdamW β(0.9, 0.95) wd 1e-2，peak **1e-5**，4000 warmup → 92% 后 cosine 衰到 peak 的 30%（vision tower 全程不冻）；BEHAVIOR 专属 LR 未单独披露
- **口径**：Standard、低分辨率 RGB，50 任务 ×10，单 generalist ckpt，2 次平均，Task-Success Score（BDDL 谓词比例，部分给分）
- **表现**：**G0.5(4 epochs) = 0.3136（headline 31.4%）**
- **同表（Table 4）G0.5 协议下 re-report**：G0.5(1ep) 0.2904 · pi0.5(4ep) **0.2626**（摘要 26.3%）· RLC 0.2605 · Comet 0.1830
  - ⚠️ 这里的 RLC 0.2605 / Comet 0.1830 是 **public-val 级**，**不是**官方 held-out（0.2599 / 0.2514），勿混

#### LEGACY RL 基线（RL-VMC / RL-Prim. / RL-Prim.Hist.）—— ❌ 与 2025 挑战不可比

- **引用**：arXiv:2403.09227（BEHAVIOR-1K 论文，CoRL'22 li23a）
- **算法**：RL-VMC = SAC 端到端视觉运动；RL-Prim.(+Hist.) = PPO + 运动规划 primitive（RRT-Connect，特权 / 传送）
- **训练**：**30,000 env steps**；batch 64；buffer 300；3 seed；LR **3e-4**（PPO γ0.99 / λ0.99 / ε0.2；SAC γ0.99 / τ0.005）
- **表现（旧 3 任务：StoreDecoration / CollectTrash / CleanTable）**：
  - Success rate：VMC 0/0/0 · Prim. 0.48/0.42/0.77 · **Prim.Hist. 0.55/0.63/0.88**
  - Q-score：VMC 0/0/0 · Prim. 0.50/0.49/0.77 · **Prim.Hist. 0.59/0.68/0.88**
- ❌ **不可比**：3 个 legacy 任务、单臂旧本体、特权抓取 primitive、旧协议 —— 与 2025 的 50 任务 / R1 Pro / 纯 IL 是不同基准

---

## 官方 starter baseline（训练配方公开，但**无 50-任务 aggregate 分数**）

> 来源：`StanfordVL/b1k-baselines` · `wensi-ai/openpi@behavior` · `wensi-ai/il_lib` · `evansh666/openvla-oft`。
> **关键事实**：官方仅发布 **per-task checkpoint**，不发布 50 任务平均 Q-score，均未上 leaderboard ——
> 因此以下是**训练配方参考，不是分数行**。

**pi0 / openpi**（config `pi0_b1k` / fork `pi05_b1k`，从 `pi0_base` / `pi05_base` 初始化）
- global batch **64**；**50k steps = declared-final**（turning_on_radio + picking_up_trash 各一个 ckpt；openpi 默认 30k）
- LR 未覆盖 → 继承默认 CosineDecay peak 2.5e-5 → 2.5e-6，warmup 1000，decay 30k，AdamW clip 1.0，EMA 0.99；action horizon 32；卡数未披露

**OpenVLA-OFT**（`openvla-7b` + LoRA r32）
- batch **4**；**max_steps 100,005，decay@50k**；lr **5e-4**；L1 回归头 + FiLM + 3 视图 + proprio，~25-action chunk
- **仅配方，无 ckpt、无分数**

**il_lib BC**（ACT / Diffusion Policy RGB(D) / DP3 / BC-RNN / WB-VIMA，均从零，action_dim = 23）
- epoch / val 驱动，**无固定 step**（`lr_cosine_steps=300000` 是 LR 调度视界，**不是** stop 点）
- lr **7e-4** cosine（warmup 1000，min 5e-6）；Adam；grad-clip 1.0；wd 默认 0（WB-VIMA / ACT 覆盖 0.1）；batch base 64、WB-VIMA / ACT 128
- **仅 WB-VIMA 发布 turning_on_radio 单任务 ckpt，其余仅配方，均无 aggregate 分数**

---

## ⚠️ 待核（出版前人工复核）

- **SimpleAI(3) / North Star(4) / Embodied Intelligence(5)** 分数：来自转载表（Comet Table 1 / RLC 博客），官方 2025 leaderboard 已迁 2026 无法直读 —— 分数高置信但**无第一方 artifact**。
- **Embodied Intelligence 的 Privileged track 归属**：反推得来（Comet "standard track" 表仅列 1–4 名），未逐字确认。
- **LEGACY RL 数值**：建议核对 arXiv:2403.09227 的 Tables 2 / A.11 / A.12 / A.13。
- **RLC 全局 batch**：无 declared-final（2048 / 16 为示例，代码默认 32）。
- **Comet 预训练卡数**：genuinely not disclosed（旧 "gpu40" 已确认为捏造）。

---

## Sources

- **官方** —— leaderboard / evaluation / baselines / dataset：`behavior.stanford.edu/challenge/` · HF：`huggingface.co/datasets/behavior-1k/2025-challenge-demos`
- **① RLC** —— arXiv:2512.06951 · GitHub `IliaLarchenko/behavior-1k-solution` · HF `IliaLarchenko/behavior_submission`
- **② Comet** —— arXiv:2512.10071（v3，赛后修订）· GitHub `mli0603/openpi-comet`
- **G0.5** —— `opengalaxea.github.io/G05/`（tech report，无 arXiv）· 前身 G0：arXiv:2509.00576
- **官方 baseline / 原始论文** —— `StanfordVL/b1k-baselines` · `wensi-ai/openpi@behavior` · `wensi-ai/il_lib` · `evansh666/openvla-oft` · BEHAVIOR-1K：arXiv:2403.09227
