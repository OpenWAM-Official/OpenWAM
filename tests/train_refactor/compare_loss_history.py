"""比对两次 seeded debug 训练的 loss-history CSV(train 重构 golden 门)。

train 重构的"与 main 行为一致"判据:固定 ``cfg.project.seed`` + ``training.debug=true``
跑出的 ``debug_loss_history.csv``,重构前后在确定性列上逐值一致。

只比对确定性列,排除依赖 wall-clock / 显存分配器的列(它们永不 bit-identical):
    steps_per_sec, mem_alloc_gb, mem_reserved_gb,
    step_peak_alloc_gb, step_peak_reserved_gb, run_peak_alloc_gb, run_peak_reserved_gb

逐值按 CSV 写出的原始字符串比对(trainer 用 ``%.10g`` 写,确定性路径下应逐字符一致),
故无需 atol —— 任何差异都暴露。latent 模式下 loss_action/loss_decoder 列名会变,
但前后同 config 故列名一致,按列名取交集即可自动对齐。

用法:
    python tests/train_refactor/compare_loss_history.py baseline.csv candidate.csv
退出码 0 = 逐值一致;1 = 有差异(打印首处)。
"""

import csv
import sys

NON_DETERMINISTIC = {
    "steps_per_sec",
    "mem_alloc_gb",
    "mem_reserved_gb",
    "step_peak_alloc_gb",
    "step_peak_reserved_gb",
    "run_peak_alloc_gb",
    "run_peak_reserved_gb",
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare(main_csv, refactor_csv):
    a = _read(main_csv)
    b = _read(refactor_csv)
    if len(a) != len(b):
        print(f"FAIL: 行数不同 main={len(a)} refactor={len(b)}")
        return 1
    if not a:
        print("FAIL: CSV 为空,没有可比对的步")
        return 1

    # 仅比对两个 CSV 都有的确定性列。模式差异(latent vs 非 latent)会改变
    # loss_action/loss_decoder 列的存在性,取交集自动对齐;打印不对称列以免静默漏比。
    only_a = [c for c in a[0] if c not in b[0] and c not in NON_DETERMINISTIC]
    only_b = [c for c in b[0] if c not in a[0] and c not in NON_DETERMINISTIC]
    if only_a or only_b:
        print(f"NOTE: 列不对称(模式差异),仅比交集。main独有={only_a} refactor独有={only_b}")
    cols = [c for c in a[0].keys() if c not in NON_DETERMINISTIC and c in b[0]]
    for i, (ra, rb) in enumerate(zip(a, b)):
        for c in cols:
            if ra.get(c) != rb.get(c):
                print(f"FAIL: 行{i} 列'{c}' 不一致: main={ra.get(c)!r} refactor={rb.get(c)!r}")
                return 1
    print(f"OK: {len(a)} 步在 {len(cols)} 个确定性列上逐值一致 ({cols})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
