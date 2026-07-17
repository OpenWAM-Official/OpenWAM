# Contributing to OpenWAM

## Development Setup

```bash
# Clone and install in editable mode
git clone https://github.com/KraHsu/OpenWAM.git
cd OpenWAM
pip install -e .

# Install dev tools
pip install ruff pytest pre-commit

# Set up pre-commit hooks (optional but recommended)
pre-commit install
```

## Common Commands

```bash
make test      # run the test suite
make lint      # check code quality with ruff
make format    # auto-format code
make check     # compile check + tests
make all       # lint + tests (full validation)
```

## Before Submitting a PR

1. Run `make all` and ensure it passes
2. Add tests for new functionality in `tests/`
3. If you changed user-visible behavior, update `README.md`
4. Keep commits focused: one logical change per commit

## Config Change Policy (All Contributors)

To keep shared configs stable and avoid environment-specific breakage:

1. Do not change existing files under `configs/**/*.yaml` unless the change is required for shared behavior (bug fix, feature, or protocol update).
2. Path-only or environment-only adjustments (dataset mounts, local OSS paths, machine-specific values) must not be committed in shared config files.
3. Prefer runtime overrides instead:
   - Hydra CLI overrides, e.g. `dataloader.dataset_dir=... dataloader.stats_path=...`
   - Local, untracked config files for personal environments
   - `sandbox/` launch scripts that inject local override arguments (see [§Sandbox](#sandbox-测试场))
4. If a config file change is unavoidable, explain the necessity in the PR description.

## Sandbox (测试场)


**用途**：

- 环境特定路径（OSS mount、本机 dataset / weight 路径）注入入口
- smoke / 100-iter 回归脚本（验证某次 refactor 没破坏，配套 W&B run id）
- 临时 ablation / 调参实验 launcher
- 跑某条数据 / 某个机型的一次性配置

**入仓动机**：让别人能复现"我这次 PR / smoke 跑了什么"，但**入仓的脚本只是
reference**，别人换机器要 fork。`sandbox/` 不是 production training entry。

**写入规则**：

| 必须 | 禁止 |
|---|---|
| 走 `scripts/train.sh`（统一 entry），享受 wandb 协议 / git SHA / run-name 自动注入 | 直接 `torchrun` 绕过协议层 |
| 环境特定值通过 env var + Hydra CLI override 传入 | 修改 `configs/**/*.yaml` 来塞本机路径 |
| 头部注释里写清楚跑过的机器 / OSS layout / W&B run | token / 密钥写脚本里 |
| 一个目录 = 一个 smoke 主题，入口固定叫 `run_smoke.sh` | 把 `sandbox/` 设成 CI / production entry |

**判断 sandbox vs scripts**：

- `scripts/` 跨机器 portable，是给所有 contributor 用的训练入口
- `sandbox/<topic>/` 只针对一台机器 / 一次性实验，本人 PR 要带、改完不一定继续维护

如果你写的脚本属于"任何人在任何机器上都该这么跑"，应该升级到 `scripts/`；
否则留在 `sandbox/`。

## Code Style

- Ruff handles linting and formatting (configured in `pyproject.toml`)
- Line length limit: 120 characters
- Import sorting: ruff isort (first-party = `open_wam`)
- `third_party/` is excluded from linting

## Dependency Management

`pyproject.toml` 里的每一个依赖**必须同时指定 lower 和 upper bound**：

```toml
# ✅ Good
"transformers>=5.5,<6"
"torch>=2.0.0,<3"
"hydra-core>=1.3,<2"

# ❌ Bad（缺 lower bound，没人知道最低能跑哪版）
"transformers<5"
"numpy<3"

# ❌ Bad（无 upper bound，上游 breaking 时炸）
"transformers>=5.5"

# ❌ Bad（精确锁定单一版本，后续升级太脆）
"transformers==5.5.0"
```

**规则**：

1. **lower bound** 是真实跑过 + 通过 `make all` 的最低版本，不是猜的最低兼容版本
2. **upper bound** 是当前能跑的最高 major/minor +1（比如 transformers 5.5 跑通就写 `<6`，不写 `<5.6` 避免每个 patch 都要升）
3. **新增 dep 时**：在 PR 里说明这个 lower bound 是怎么验证的（CI 跑过 / 本地装这个版本测过）
4. **升 upper bound 时**：本地装最新版本跑一遍 `make all` 再升，不要"大概应该兼容"
5. **不要使用 `==` 精确锁定**，除非有明确的 reproducibility 要求（这种情况记到 PR 描述）

**为什么这么严**：

我们已经踩过 transformers 5.5+ 才有 `Qwen3_5ForConditionalGeneration` 的坑（早期版本 import 直接 NotFound）。原 `pyproject.toml` 写 `transformers<5` 等于允许装 4.x，新 contributor 一装 4.x 跑 Qwen 直接挂。完整区间 pin 是这种问题的唯一根治方案。

**新 PR 引入的新 dep 必须按上面规则双向 pin**。`pyproject.toml` 历史上的 hygiene 债已于 2026-04-30 一次性 audit 完成（27 个 dep 全部双向 pin）。

## Follow-ups & Engineering Debt


- [`plans/`](plans/)：前瞻设计 / 架构规划

每条 follow-up 必须带：现状、目标、**为什么没做**（blockers）、**什么时候重新评估**（触发器）、触发后动作。**不要把 follow_ups.md 当 idea dump**，会死。


## Project Structure

- `open_wam/` - main package (all new code goes here)
- `scripts/` - Hydra entrypoints (train, infer, eval)
- `configs/` - Hydra config groups
- `tests/` - pytest test suite
- `third_party/` - vendored dependencies (do not modify unless necessary)

## Adding a New Component

### New dataset
1. Create a reader inheriting from `openwam.dataloader.bases.BaseDataset` (single-bucket LeRobot v3 readers subclass `LeRobotV3Reader`)
2. Register it in `open_wam/data/registry.py`
3. Add a config in `configs/data/my_dataset.yaml`

### New architecture
1. Create `open_wam/models/architectures/my_arch.py` inheriting from `BaseWAMArchitecture`
2. Register with `@register_architecture("my_arch")`
3. Add a config in `configs/model/architecture/my_arch.yaml`

### New evaluator
1. Create `open_wam/evaluation/my_evaluator.py` inheriting from `BaseEvaluator`
2. Register in `open_wam/evaluation/registry.py`
3. Add a config in `configs/eval/my_eval.yaml`
