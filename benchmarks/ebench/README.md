# EBench × OpenWAM 评测桥

把 OpenWAM(EBench pretrain-SFT ckpt)接入 [EBench](https://github.com/InternRobotics/EBench)
(GenManip / Isaac Sim,lift2 双臂移动机器人)。与 robotwin / robocasa365 不同,EBench 的
**策略侧本身就是客户端**(`genmanip_client.EvalClient` 轮询 sim 服务器),因此本桥是一个独立
驱动进程,不需要 behavior 式的双跳协议服务器:

```
GenManip eval server(Isaac Sim 本地 :8087 或官方在线 endpoint)
    ▲ EvalClient(pickle wire;obs 下行 / action dict 上行)
openwam2ebench_interface.py  —— 每个 worker 一个进程
    ▼ WSPolicyClient(JSON-WS)
OpenWAM policy server(:8848;每 worker 一个,executor 有状态)
```

## Files

| 文件 | 作用 |
| --- | --- |
| `openwam2ebench_interface.py` | 驱动:EvalClient 循环 ↔ OpenWAM server |
| `prompt_template.py` | 训练模板逐字节镜像(回归测试锁定) |
| `policy_config.yml` | 连接/动作配置模板 |
| `single_eval.sh` / `multi_eval.sh` | 单/多 worker 启动脚本 |
| `mock_genmanip_server.py` | 无 Isaac Sim 的离线 mock(真实数据回放 + wire 契约校验) |
| `../utils/action_conversion.py` (EBench 段) | 纯 numpy 转换,与 dataloader 逐字节镜像 |
| `../../tests/benchmarks/test_ebench_bridge.py` | 离线测试(镜像/round-trip/wire 契约) |

## 动作空间契约

模型输出 unified 80-D → OpenWAM server 端 `_UnifyAwareNormalizer` gather 回 raw-23 并反归一化
(物理单位)→ 本桥仅做表示转换:

| raw-23 | 语义 | 上行编码 |
| --- | --- | --- |
| `[0:3) [3:9) [9]` | 左臂 xyz + rot6d + 标量夹爪 | `(pos3, quat_wxyz4, [g,g])`,rot6d→quat,g clip [0,0.044] |
| `[10:20)` | 右臂,同上 | 第二个 arm tuple |
| `[20:23)` | base,**由训练的 `base_action_source` 决定** | 见下 |

- `--base-mode delta`(默认,对应 `dataloader.base_action_source=delta`):
  `base_motion=[dx_m, dy_m, dyaw_deg]`,`base_is_rel=true`。GenManip 每步 clip
  ±0.015 m / ±1°(demo 遵守的同一约束),内部 deg2rad。**yaw 保持度数,勿换算。**
- `--base-mode cumulative`:绝对 `base_motion=[x_m, y_m, yaw_deg]`,`base_is_rel=false`
  (服务器对 index 2 做 `np.deg2rad`)。
- 臂端 `control_type="ee_pose"`、`is_rel=false`(绝对位姿,服务器端 cuRobo 每步 IK,
  与训练 FK 同一 per-arm base frame)。pos/quat 必须是 **Python list**(服务器做
  `position + orientation` 列表拼接,ndarray 会广播相加然后崩)。
- ⚠️ 官方三条 baseline(π0/X-VLA/InternVLA-A1)全部走 `joint_position`;`ee_pose`
  路径无先例,IK 失败会静默保持当前关节——上线前务必先本地 sim 验证。

## 观测契约

| GenManip obs | → OpenWAM 请求 |
| --- | --- |
| `video.overlook_camera_view` | `images.head_camera`(必需;server 端按 ckpt config 拼 384×320 L 形) |
| `video.left/right_camera_view` | `images.left/right_wrist_camera` |
| `instruction` | `prompt_template.format_prompt_for_inference()` 包装后作为 `prompt` |
| `state.ee_pose/gripper/base` | RAW-23 proprio(`ebench_obs_to_raw23`;base 经 `ebench_render_state_base` 渲染进指令空间——delta 模式为相邻步测量差分,yaw wrap 后 rad→deg;server 端归一化) |
| `obs["reset"]==True` | south `reset`(清 action buffer/ensemble)+ 桥内 prev_base 清零 + 重读 instruction |

仅单步 `step()`:`/step_chunk` 只回末帧 obs,与 OpenWAM executor "一 obs 一 action" 不兼容。
replan 频率是 server 端的事(`execute_horizon`,默认贪心吃满 32-step chunk)。

## 本地评测(需 Isaac Sim 4.1.0 机器)

```bash
# sim 机(pre-Blackwell GPU;pip isaacsim==4.1.0 + cuRobo;EBench-Assets ~11.4GB)
python ray_eval_server.py --host 0.0.0.0 --port 8087 --no_save_process
gmp submit ebench/generalist/val_train --run_id smoke1

# 推理机
bash scripts/deploy.sh --ckpt-dir <ebench_ckpt> --port 8848
EBENCH_PYTHON=<genmanip-client env python> bash benchmarks/ebench/single_eval.sh \
    --url http://<sim-host>:8087 --run-id smoke1
```

`test_mini` 乃至 `test` split 的任务配置与资产都是公开的,本地可完整复现;只有官方在线
提交(仅收 generalist track,跑 Test-Mini)计入 leaderboard 并产出多轴诊断报告。

## 官方在线评测

权重不上传;模型跑在自己机器上,与本地评测用**同一套桥**,仅 north 端点不同:

```bash
gmp online submit --base_url https://internrobotics.shlab.org.cn/eval --token $TOK \
    --benchmark_set ebench_generalist ...   # 返回 task_id + endpoint
EBENCH_PYTHON=... bash benchmarks/ebench/single_eval.sh \
    --url "$ENDPOINT" --token "$TOK" --run-id "$TASK_ID"
```

限制:≤16 并发 worker、10 分钟不活动断线、失败可用同 task_id 续跑。

## 离线验证(无 Isaac Sim)

```bash
# 终端 1:mock GenManip(真实数据回放 + 动作契约校验)
python benchmarks/ebench/mock_genmanip_server.py \
    --dataset-dir <EBench-Dataset> --bucket simple_pnp/task1 --episodes 2 --steps-per-episode 8
# 终端 2:真实 OpenWAM server(smoke ckpt 即可)
bash scripts/deploy.sh --ckpt-dir <ckpt> --port 8848
# 终端 3:真实桥
EBENCH_PYTHON=... bash benchmarks/ebench/single_eval.sh --url http://127.0.0.1:8087
```

纯离线单测(无服务、无数据):`pytest tests/benchmarks/test_ebench_bridge.py`,随
`make test` 运行。

## 已知限制 / 状态

- mock 闭环验证 wire 编码/转换/推理链路,且以 HTTP 500 强制契约(坏桥 fail 而非静默过),
  但**不产生分数**、无 IK/物理/提前终止;真实闭环(Isaac Sim)未在本仓验证。
- `--base-mode` 必须与 ckpt 训练配置一致:ckpt 目录可达时**务必**传
  `--ckpt-config <ckpt>/config.yaml` 硬校验(错配会静默产出灾难性 base 动作);
  不可达时桥打 UNVERIFIED 警告。
- 配置可放 `--config benchmarks/ebench/policy_config.yml`(CLI 优先)。
- 吞吐:794 实例 generalist ≈ 1.79M sim 步;每次 replan 一次视频模型推理,预算见
  `openwam/deploy` 的 `execute_horizon`。

## Isaac Sim 实测前检查单(gate-3 输出)

1. 用最终 ckpt 跑 `--ckpt-config` 校验;确认 stats 含 `ebench`、normalize=min-max、
   `base_action_source` 与 `--base-mode` 一致、camera layout=overlook/left/right。
2. 单 worker 单 episode 起步:核对 reset/prompt/episode_id/timestep、base 首步零差分;
   注入一次 north 断连,确认旧动作未进入新 episode 且重连首帧 `reset=True`。
3. 观察 GenManip 日志的 IK 成功率(cuRobo `None→hold` 是静默的);对比 commanded EE pose
   与下一步 `state.ee_pose`。
4. 检查 invalid-state termination_reason 计数(arm/gripper/base 范围与单步跳变)。
5. 三路图像目检 uint8 RGB(红/蓝物体对比训练视频与 south 预处理输出)。
6. 再扩 2 worker:各自独立 south server/端口/结果记录。
7. 在线评测前:endpoint/token/`run_id==task_id`、≤16 workers、首次推理编译不要撞
   10 分钟 inactivity 断线(先本地 warmup)。
