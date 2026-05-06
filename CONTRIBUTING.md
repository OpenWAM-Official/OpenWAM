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
5. Keep commits focused: one logical change per commit

## Config Change Policy (All Contributors)

To keep shared configs stable and avoid environment-specific breakage:

1. Do not change existing files under `configs/**/*.yaml` unless the change is required for shared behavior (bug fix, feature, or protocol update).
2. Path-only or environment-only adjustments (dataset mounts, local OSS paths, machine-specific values) must not be committed in shared config files.
3. Prefer runtime overrides instead:
   - Hydra CLI overrides, e.g. `dataloader.dataset_dir=... dataloader.stats_path=...`
   - Local, untracked config files for personal environments
   - `sandbox/` launch scripts that inject local override arguments (see [§Sandbox](#sandbox-测试场))

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
| 走 `scripts/train.sh`（或对应统一 entry），享受 wandb 协议 / git SHA / run-name 自动注入 | 直接 `torchrun` 绕过协议层 |
| 环境特定值通过 env var + Hydra CLI override 传入 | 修改 `configs/**/*.yaml` 来塞本机路径 |
| 头部注释里写清楚跑过的机器 / OSS layout / W&B run | token / 密钥写脚本里 |
| 一个目录 = 一个 smoke 主题，入口固定叫 `run_smoke.sh` | 把 `sandbox/` 设成 CI / production entry |

**判断 sandbox vs scripts**：

- `scripts/` 跨机器 portable，是给所有 contributor 用的训练入口
- `sandbox/<topic>/` 只针对一台机器 / 一次性实验，本人 PR 要带、改完不一定继续维护

如果你写的脚本属于"任何人在任何机器上都该这么跑"，应该升级到 `scripts/`；
否则留在 `sandbox/`。

## Updating the CHANGELOG


### When to update

- ✅ New feature, refactor, fix, perf change, dataset / config / interface change
- ✅ Anything you'd want a future engineer to find when bisecting an issue
- ❌ Pure typo / comment / formatting fixes
- ❌ Docs-only changes inside `docs/`
- ❌ Test-only changes that don't modify production behavior

### Entry format

Group entries by date (`## YYYY-MM-DD: <短主题>`) in reverse chronological
order. Within a date, group bullets by stage:

| Section | When |
|---|---|
| `### 新增` | 新文件 / 新接口 / 新 config |
| `### 变更` | 现有行为修改、refactor、interface 重命名 |
| `### 修复` | bug fix |
| `### 性能` | benchmark 数字变化（必须给 before / after）|
| `### 验收` | 这次提交跑了什么测试 / 训练，附 W&B / pytest 输出 |
| `### 已知 Gap` | 留给后续 PR 的 TODO，写明触发条件 |

Within each section, each bullet should:

1. Lead with a `**type(scope)**` tag (`feat(framework)`, `fix(fsdp)`,
   `refactor(dataloader)`, ...)
2. Cite **specific file paths and line ranges** (`openwam/.../foo.py:464-477`)
   so readers can navigate without searching
3. Explain **why** non-obviously — what the previous behavior was, what
   constraint forced the change, what tradeoff was taken
4. Quote concrete numbers (loss, throughput, memory, sample count) over
   adjectives ("faster", "smaller")
5. Link external artifacts when they prove the claim — W&B run IDs, PR /
   issue numbers, golden test names

Example (good):

> `**fix(loader)**: Handle empty batches in openwam/dataloader/example.py:42-57. Explain the root cause, the public behavior change, and the regression test that verifies it.`

Example (bad — 不要这样写):

> `修了一些 bug，性能更好了。`

### Where the entry goes

- 提交分支上的工作完成 + 测试通过后，把当次 PR 的 changelog 段写到
- 该段日期使用 commit / PR 即将合并日（不一定是最早开发日），保证倒序时序正确。
- 如果同一天已经有别的 entry，**附加一个新二级标题**（`## YYYY-MM-DD: <主题二>`），
  不要合并到别人的 entry 里，避免 git diff 串味。

### Linking back from CHANGELOG

CHANGELOG entries通常引用其他文档作为权威细节出处：

- Protocol / spec：写到 `docs/<topic>.md`，CHANGELOG 提一行带链接
- 性能 / 训练曲线：留 W&B run id / link
- 数据格式 / dataset layout：写到 `docs/dataloader.md` 或对应 dataset
  README，CHANGELOG 引用

CHANGELOG 本身**不是** spec —— 它说"哪天改了什么、为什么"，详细规范在 `docs/`。

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
1. Create `open_wam/data/my_dataset.py` inheriting from `BaseActionDataset`
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
