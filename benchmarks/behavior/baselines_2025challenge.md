# BEHAVIOR-1K 2025 Challenge — Baselines & Training-Config Survey

> 调研交付物(对应集成验收 §三:① 相关工作训练步数/batch 配置;② 已有 baseline 得分),
> 供后续微调测试与论文图表参照。
>
> **方法**:多 subagent fan-out 联网检索 → 5 路独立搜索收敛 → 第二个独立 agent 对 load-bearing
> 数字逐条重新 fetch 一手源交叉核验。置信度标注:✅ 确证(≥2 个一级来源)/ ⚠️ 单源待核 / ❌ 未找到。
> 所有数字均以官方 leaderboard、两位获奖者 arXiv 论文、官方 baselines/dataset 页为准。

## 0. 一句话总览

2025 BEHAVIOR Challenge(NeurIPS 2025 竞赛赛道;50 个长程家务任务,10,000 条遥操作 demo,机器人
**Galaxea R1 Pro**)的**排名指标是 Q-score(部分成功的子目标比例,给部分分,非二元成功率)**。
**第一名 Robot Learning Collective(Ilia Larchenko 独立队),held-out test Q-score 0.2599,方案基于
Physical Intelligence 的 π0.5(Pi0.5)VLA + flow matching**;第二名 NVIDIA "Comet"(同样 π0.5)
held-out Q 0.2514。官方提供的 baseline 是 **π0/openpi、OpenVLA(-OFT)、ACT、Diffusion Policy、
BC-RNN、WB-VIMA**,但**官方未公布这些 baseline 的 50 任务聚合得分**。

> 与本项目的直接关联:两位 top 选手都用 **π0.5 + 8×H200**;第一名的**动作空间 23D
> (3 base 速度 + 4 trunk + 7+1 左臂/夹爪 + 7+1 右臂/夹爪)与本仓库 BEHAVIOR dataloader 的
> `action[23]` schema 完全一致**,可直接作为微调配置与 sanity-check 的参照。

---

## 1. Baseline 得分

### 1a. 竞赛排行榜(官方;排名指标 = Held-out test Q-score)

**口径**:50 任务 / 每任务 200 训练 demo;评测 = 10 个 public-validation 实例 + 10 个 held-out(private)
实例;仅 top-5 跑 held-out 测试集。Q-score = 各任务"已满足 BDDL 目标谓词数 / 总目标谓词数"在 50 任务
上的平均;Full Success = 二元全成功(**不用于排名**)。赛道:**Standard**(仅 RGB-D+seg+proprio)/
**Privileged**(可查询仿真器特权信息)。

| 排名 | 队伍 | 所属 | 赛道 | Full Success (pub-val / test) | **Q-score (pub-val / test)** | 置信度 |
|---|---|---|---|---|---|---|
| 1 | **Robot Learning Collective**(Larchenko, Zarin, Karnatak)| Independent | Standard | 0.1120 / 0.1240 | 0.2605 / **0.2599** | ✅ |
| 2 | **Comet**(π0.5)| NVIDIA Research | Standard | 0.1440 / 0.1140 | 0.1830 / **0.2514** | ✅ |
| 3 | SimpleAI Robot | Beijing Simple AI | Standard | 0.1400 / 0.1080 | 0.1943 / **0.1591** | ✅ |
| 4 | The North Star | Huawei CRI EAI | Standard | 0.1280 / 0.0760 | 0.1702 / **0.1204** | ✅ |
| 5 | Embodied Intelligence | Independent | Privileged | 0.0620 / 0.0520 | 0.1110 / **0.0947** | ✅ |
| 6–18 | RAPPER(GIST)、RACΞL(CMU)、Merlin Labs、"ACT"(Xiamen, Q_val 0.0037)… | 各异 | 多 Standard | → 0.0 | → 0.0 | ⚠️ 单源(榜单)|

**反直觉点(已交叉核验)**:第二名的 **public-val full-success(0.1440)高于第一名(0.1120)**,
但排名按 **held-out test Q-score**(第一 0.2599 > 第二 0.2514),故第一名夺冠。注意 0.1440 vs 0.1120
是 *public-val full-success* 指标,不是 held-out。

规模:**18 支队伍(4 国:美/中/加/韩,⚠️ 国别单源)**;每赛道各取 top-3;奖金 $1,000/$500/$300。

### 1b. 官方 Baseline 方法(官方未公布 50 任务聚合得分)

| Baseline | 类型 | 官方聚合得分? | 置信度 |
|---|---|---|---|
| **π0 / openpi**(旗舰)| VLA(flow matching)| ❌ 仅给单任务 checkpoint(turning_on_radio / picking_up_trash, 50k-step)| ✅ 方法 |
| **OpenVLA / OpenVLA-OFT(+)** | VLA | ❌ | ✅ 方法 |
| **ACT** | BC(action chunking)| ❌(榜单第 13 名一支参赛队恰名为 "ACT", Q_val 0.0037,但那是**参赛提交**,非官方 baseline)| ✅ 方法 |
| **Diffusion Policy**(RGB(D) + 3D)| BC | ❌ | ✅ 方法 |
| **BC-RNN** | BC | ❌ | ✅ 方法 |
| **WB-VIMA** | BC(whole-body)| ❌ 仅 turning_on_radio checkpoint | ✅ 方法 |

> 措辞说明(交叉核验):官方 index/CFP 宣传清单写的是 **"OpenVLA"**,而 baselines 教程页给出的是
> **OpenVLA-OFT / OpenVLA-OFT+** 的微调配方——两者并存。
>
> ❌ **2025 Challenge 无 RL/PPO/SAC 官方 baseline**;若论文需要 RL 对照,只能引 **2024 原始 BEHAVIOR-1K
> 论文(arXiv:2403.09227)**,但其口径完全不同(3 个活动、用 sticky/assistive 抓取原语,不可与 2025 比):
> RL-Prim.Hist.(PPO+原语+历史)在 StoreDecoration/CollectTrash/CleanTable 上为 0.55 / 0.63 / 0.88(✅ 论文 Table 2)。

---

## 2. 训练配置

| 方法 | base | steps | batch | lr / optimizer | 数据量 | 硬件 | 置信度 |
|---|---|---|---|---|---|---|---|
| **官方 π0 baseline**（`pi0_b1k`，从 `pi0_base` 初始化）| π0 | **50,000** | **64** | LR ❌(藏在 openpi config)| 单/少任务(教程 turning_on_radio）| ❌ 无官方推荐;仅 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（暗示单张大显存 GPU）| ✅ steps/batch；LR ❌ |
| **官方 OpenVLA-OFT baseline** | OpenVLA | **~100,005**(decay@50k）| **4** | **5e-4**（唯一明确给 LR 的官方 baseline）| LoRA rank 32, 3 图, 25-act chunk | ❌ 未明示 | ✅ |
| **第 1 名 Robot Learning Collective** | **π0.5**（action expert Gemma 300M, ~311M, JAX/FSDP）| README 示例 `num_train_steps=200000`;论文按时长（"多任务 15 天 + 每组 ~1 周","≈2 epochs"）| 多卡 **2048** / 单卡 **16**（README 示例）| LR ❌ | **10,000 demos / 50 任务 / 1,200+ 小时**（=200/任务）;RGB-only 224×224 子集 ~260GB | **8× H200**（训练 FSDP）;推理单张 **RTX 4090**;评测扩 20× 4090 | 硬件 ✅;steps/batch ⚠️（示例非最终）;LR ❌ |
| **第 2 名 Comet** | **π0.5**（JAX）| 多任务预训练 **50k**;单任务 SFT **15k–20k** | per-device **64** | 预训练 **2.5e-5**，SFT/RFT **2.5e-6**，**cosine**（唯一给全套 LR 的竞品）| 官方 10k + 自采 **~3.6K** 规划器/离线RL 轨迹;RFT 3 轮 | **8× NVIDIA H200**（SFT/RFT）| ⚠️ 单源（其论文，为权威一手源）|

**第 1 名补充(⚠️ 多为单源即其论文/README)**:两阶段(50 任务多任务 → 拆 4 个 task-group 专用
checkpoint,最终提交 = 4 个 ckpt);核心创新 = correlated noise for flow matching(β=0.5)、learnable
mixed-layer attention、System-2 stage tracking、可学习任务嵌入(替代语言 → 严格说是 vision-action 非
VLA)、multi-sample flow matching、推理期 gripper 启发式纠正;**动作空间 23D**(3 base 速度 + 4 trunk +
7+1 左 + 7+1 右),horizon 30,3 路相机(头+双腕,224×224)。预算 ~$13k(Nebius 赞助 $10k)。

**数据集(官方,✅ 多源)**:10,000 条 100% 人类遥操作 demo / 50 任务 / 200 每任务 / **1,200+ 小时**;
平均轨迹 397s(6.6 min);**LeRobot 格式,全量 ~1.5TB**;本体 **Galaxea R1 Pro**(轮式双臂人形);
采集系统 JoyLo;HF `behavior-1k/2025-challenge-demos`。
⚠️ **数据坑(对 dataloader 项目重要)**:dataset 页注明 parquet 中 robot state 的 **joint-efforts 字段是
错的、勿用于训练**,官方下版会移除。

---

## 3. 数据缺口 / 存疑(已逐条交叉核验)

- ❌ **官方 baseline(π0/ACT/DP/BC-RNN/WB-VIMA/OpenVLA)的 50 任务聚合成功率/Q-score:未找到**。官方只给
  配方与个别单任务 checkpoint;榜单上的分都是**参赛提交**,非官方 baseline 基准线。若论文需"baseline
  floor",可引榜单第 13 名参赛队 "ACT"(Q_val 0.0037)作 ACT 类弱下限,但须注明是参赛实现而非官方 baseline。
- ❌ **π0 官方 baseline 的 learning rate / optimizer / weight decay:未找到**(只给 batch=64、steps=50k)。
  两位 top 选手中只有 Comet 给了完整 LR。
- ❌ **官方硬件推荐:未找到**(无 A100/H100 官方建议);能确证的只有数据集体量(~1.5TB)与选手自报 8×H200。
- ⚠️ **Comet 摘要的 "validation Q-score 0.345"** 是 **赛后/不同 validation split** 的更高分,**不是 held-out
  test**;竞赛口径权威数字是 held-out **0.2514**(本表统一用 0.2514)。
- ⚠️ **第 1 名 README 的 batch=2048 / steps=200000 是"可运行示例"而非声明的最终超参**(论文改用墙钟时间
  描述);引用请标注为示例值。
- ⚠️ **deadline**:原为 2025-11-13,官方延期 24h 至 **2025-11-16 11:59PM AoE**;榜单各队提交日 20251114–20251117。
- ❌ **应忽略的假信息**:某 AI 聚合站(blockchain.news/ainews)称"领先队伍平均成功率超 70%"——与所有一手源
  直接矛盾(最高 held-out Q ≈ 0.26、最高 full-success ≈ 0.124),**判定为 AI 幻觉,勿引用**。

---

## 4. Sources

**官方（behavior.stanford.edu）**
- 总览/指标:https://behavior.stanford.edu/challenge/index.html
- Baselines（π0/OpenVLA-OFT/il_lib 配方):https://behavior.stanford.edu/challenge/baselines.html
- 评测与规则（指标定义/赛道/实例数):https://behavior.stanford.edu/challenge/evaluation.html
- 排行榜 + Q-score 定义:https://behavior.stanford.edu/challenge/leaderboard.html
- Call for Participation（时间线/规模/奖金):https://behavior.stanford.edu/challenge/call_for_participation.html
- 数据集统计:https://behavior.stanford.edu/challenge/dataset.html
- HF 数据集:https://huggingface.co/datasets/behavior-1k/2025-challenge-demos

**第 1 名（Robot Learning Collective）**
- arXiv 2512.06951(技术报告):https://arxiv.org/abs/2512.06951 ・ HTML https://arxiv.org/html/2512.06951v2
- Blog:https://robot-learning-collective.github.io/winning-behavior-1k-challenge.html
- GitHub:https://github.com/IliaLarchenko/behavior-1k-solution ・ HF 权重:https://huggingface.co/IliaLarchenko/behavior_submission

**第 2 名（Comet / NVIDIA）**
- arXiv 2512.10071(Openpi Comet):https://arxiv.org/abs/2512.10071 ・ HTML https://arxiv.org/html/2512.10071v1
- GitHub:https://github.com/mli0603/openpi-comet

**官方 baseline 仓库 / 原始论文**
- StanfordVL/b1k-baselines:https://github.com/StanfordVL/b1k-baselines
- StanfordVL/BEHAVIOR-1K:https://github.com/StanfordVL/BEHAVIOR-1K
- 2024 原始基准论文(RL baselines, Table 2):https://arxiv.org/abs/2403.09227

**赛事/新闻**
- Stanford HAI:https://hai.stanford.edu/news/behavior-challenge-charts-the-way-forward-for-domestic-robotics
