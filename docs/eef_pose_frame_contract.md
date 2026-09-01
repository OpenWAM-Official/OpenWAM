# Active-mixture EEF pose-frame contract

## Decision

The real-robot pose slots in the shared 80-D action/proprio space mean:

> A terminal-arm frame rigidly attached to the arm chain and independent of
> gripper/finger articulation, expressed in the dataset's robot-base frame.

The machine-readable name is
`rigid_terminal_arm_frame_independent_of_gripper_motion` (see
`openwam.dataloader.utils.eef.EEF_POSE_FRAME_CONTRACT`).

This is deliberately not called a universal literal `flange_pose` or
`tcp_pose`. Across embodiments the published terminal endpoint can be a flange,
wrist-yaw link, hand base, last arm link, gripper mount, or a fixed tool frame.
The contract excludes a point whose transform changes with gripper
articulation, but it does not claim that every fixed point is upstream of every
possible static TCP. It aligns the endpoint *category*; it does not assert that
different robots share the same local origin, local-axis convention, or a
common base origin. Achieving that stronger equivalence requires
per-embodiment calibrated rigid transforms, which several public releases do
not provide.

## Active datasets

| Dataset | Reader source used for pose | Physical endpoint | Decision |
|---|---|---|---|
| AgiBotWorld-Beta | `action.ee_base`, `observation.state.ee_base` | Axis-7 flange / arm end | Keep. This is the release-proven endpoint available across gripper and dex-hand embodiments. |
| InternData-A1 | `*.ee_to_robot_pose` | Per-model EE attachment/controller link (hand base, last arm link, or gripper mount) | Keep. Do not use the downstream native `*.tcp_to_robot_pose`. |
| RoboCOIN | `eef_sim_pose_action`, `eef_sim_pose_state` | Published simulation-FK terminal point; exact link varies by robot | Keep under the broad terminal-arm contract. Do not add a blanket TCP offset. |
| DROID | `other_information.action_wrist_pose`; `other_information.observation_gripper_pose6d` + `state[6]` | Commanded wrist and achieved rigid gripper-mount frame | Keep. The stats contract rejects stats generated from the moving TCP streams. |

## AgiBotWorld-Beta

The official Beta schema calls `state/end/*` the robot flange and gives actions
the same semantics. OpenWAM therefore uses `ee_base` directly. No universal
task-TCP column or calibration is published across the release's gripper and
dexterous-hand embodiments, so applying one guessed flange-to-TCP offset would
be incorrect.

Primary schema: [AgiBotWorld-Beta README](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/blob/main/README.md#L357-L400).

## InternData-A1

The public release contains both EE and TCP columns. Their orientations agree,
while TCP is a fixed local translation downstream of the selected EE link.
OpenWAM retains `*_ee_to_robot_pose` because it represents the
simulator-controlled terminal-arm endpoint. It also uses `*_to_robot_pose`
rather than per-arm-base fields so both arms share one base frame.

| Variant | Selected EE link | Published EE-to-TCP translation |
|---|---|---:|
| Franka + Panda Hand | `panda_hand` (hand base, not literal flange) | local +z 0.095 m |
| Franka + Robotiq | `panda_link8` (end-of-arm/flange) | local +z 0.145 m |
| ARX Lift-2 | `link6` (last arm link / gripper base) | local +x 0.16157 m |
| Genie-1 | `arm_{l,r}_end_link` (last arm link / gripper mount) | local +z 0.22 m |
| Split Aloha/Piper | `link6` (last arm link / gripper base) | local +z 0.135 m |

Release converters: [Panda](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_franka_a1.py#L315-L421),
[Franka+Robotiq](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_frankarobotiq_a1.py#L311-L420),
[Lift-2](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_lift2_a1.py#L382-L487),
[Genie-1](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_genie1_a1.py#L384-L505), and
[Split Aloha](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_split_aloha_a1.py#L382-L487).

## RoboCOIN

RoboCOIN guarantees unified base axes/origins for `eef_sim_pose_*`, but its
public release does not identify one common target link and the FK field
generator is not published. The best-supported public endpoint mapping is:

| `robot_type` | Best-supported endpoint |
|---|---|
| `agilex_magic`, `agilex_magic_decoupled`, `aloha` | Piper `link6 == gripper_base` |
| `airbot_mmk2`, `discover_mmk2` | AIRBOT Play v3 `link6` |
| `g1dex3`, `g1ego`, `g1high` | Unitree G1 `wrist_yaw_link` |
| `realman_rmc_aidal` | RM75B `link7` (jaw base is downstream) |
| `ruantong_a2d` | AgiBot-G1 uses `arm_end_link == gripper_base`; other families are not publicly identifiable |
| `ai2_alphabot2`, `alphabot2`, `galaxea_r1_lite`, `leju`, `yinhe` | Exact link is not established by public artifacts |

No blanket task-TCP conversion is safe across these types. Such a conversion
would require per-embodiment, and sometimes per-family, calibration.

Primary definition and models: [RoboCOIN](https://github.com/FlagOpen/RoboCOIN/blob/b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd/README.md#eef_sim_pose-state--eef_sim_pose-action),
[Piper](https://github.com/agilexrobotics/piper_ros/blob/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/src/piper_description/urdf/piper_description.urdf),
[Unitree G1](https://github.com/unitreerobotics/xr_teleoperate/blob/845b25a32f7febedf220e830952a7134897adb9d/assets/g1/g1_body29_hand14.urdf),
[AIRBOT Play v3](https://github.com/TATP-233/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/urdf/airbot_play_v3_gripper.urdf),
[RealMan RM75B/AIDA](https://github.com/RealManRobot/URDF-to-XACRO/blob/ccacc05c1cf8fe5adf05c5f1de5d53b85f286558/rm_Lifting_robot_75B_jaw_description.zip), and
[AgiBot G1](https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/1eb3b68b740f87fb369b0146ee53f3bb3da6b0d0/G1_omnipicker/G1_omnipicker.urdf).

## DROID

DROID's converted rows contain physically distinct wrist/mount and task-TCP
streams. The TCP offset varies with gripper state, so it is not one fixed
terminal-arm transform. The reader therefore uses `action_wrist_pose` and
`observation_gripper_pose6d + state[6]`.

Normalization statistics must be generated from the same reader population
and source columns. The DROID stats contract records the pose-frame semantics
and exact source columns, rejects legacy TCP-based statistics, and keeps rot6d
statistics identity-pinned.

Converter schema: [RoboInter DROID converter](https://github.com/InternRobotics/RoboInter/blob/main/RoboInterData/convert_to_lerobot/convert_droid_to_lerobot_anno_fast.py).
