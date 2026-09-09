# RoboDojo score verification for OpenWAM-α

This directory answers [issue #23](https://github.com/OpenWAM-Official/OpenWAM/issues/23),
which reported that averaging the 12 per-task Generalization cells published for
OpenWAM-α gives 14.83 while the reported Gen-Std is 25.56.

Both numbers are correct. They are two different views of the same evaluation:
14.83 is the merged 50-episode Generalization result, 25.56 is the standard-half
result. Everything below is reproducible from the files in this directory.

The issue also observed that OpenWAM-α does not line up with the other models on
the leaderboard. **That part is right**, and it is a real defect — but it lives in
the leaderboard's per-task export, not in OpenWAM-α's scores. Section 5 covers it.

## Contents

| File | What it is |
|---|---|
| `openwam_robodojo_per_seed.json` | Per-seed results for all 42 tasks, plus the standard and random halves of the 12 Generalization tasks |
| `verify_robodojo_scores.py` | Recomputes every published number from that file; `--leaderboard` also audits the live bundle |

```bash
python verify_robodojo_scores.py                # checks A-D, offline, no dependencies
python verify_robodojo_scores.py --leaderboard  # adds check E against the live leaderboard
```

Exit code is 0 only if every check passes.

## 1. How RoboDojo evaluates a Generalization task

Each of the 12 Generalization tasks is a **paired** task: 25 episodes on the
standard layout plus 25 on the `_random` layout, 50 in total. From the
benchmark's own aggregation script,
[`scripts/internal/summarize_result.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/scripts/internal/summarize_result.py):

> A task `X` that also has a sibling task `X_random` is a "paired" task: we take
> the first 25 entries of `X` and the first 25 of `X_random` (matched by policy +
> embodiment + seed) and merge them into 50.
>
> `_random` tasks are not reported on their own; they only feed their base.
>
> The Generalization section additionally lists each of the 12 tasks as separate
> 25-episode standard and `_random` rows. **This is extra output only and does not
> change the merged 50-episode scores used elsewhere.**

So one paired task yields three numbers, and the leaderboard publishes all three
in different places:

| View | What it covers | OpenWAM-α SR |
|---|---|---|
| **Generalization** (per-task cell, dimension score) | merged 50 episodes | **14.83** |
| **Gen-Std** | standard half only | **25.56** |
| **Gen-Rand** | random half only | **4.11** |

They are tied together by construction: `(25.56 + 4.11) / 2 = 14.83`.

## 2. The per-task cells are merged values

For every one of the 12 tasks, at every seed, the published cell is exactly the
mean of the two halves. Standard / random / published, per seed:

| Task | Standard | Random | (S+R)/2 | Published cell |
|---|---|---|---|---|
| stack_bowls | 88, 92, 84 | 4, 0, 4 | 46, 46, 44 | 46, 46, 44 |
| push_T | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| pack_objects_into_box | 4, 12, 12 | 0, 0, 0 | 2, 6, 6 | 2, 6, 6 |
| fold_clothes | 92, 80, 88 | 16, 20, 36 | 54, 50, 62 | 54, 50, 62 |
| hang_mugs | 16, 12, 4 | 0, 0, 0 | 8, 6, 2 | 8, 6, 2 |
| sweep_blocks | 0, 8, 0 | 0, 0, 0 | 0, 4, 0 | 0, 4, 0 |
| pour_liquid_into_cup | 56, 64, 44 | 16, 16, 12 | 36, 40, 28 | 36, 40, 28 |
| make_toast | 0, 8, 4 | 0, 4, 0 | 0, 6, 2 | 0, 6, 2 |
| arrange_largest_number | 0, 0, 4 | 0, 0, 0 | 0, 0, 2 | 0, 0, 2 |
| sort_nesting_dolls_by_size | 8, 8, 12 | 0, 0, 0 | 4, 4, 6 | 4, 4, 6 |
| store_laptop_and_headphones | 28, 36, 12 | 4, 4, 12 | 16, 20, 12 | 16, 20, 12 |
| stack_blocks | 16, 8, 20 | 0, 0, 0 | 8, 4, 10 | 8, 4, 10 |

36 of 36 values reproduce. The 12 cells quoted in the issue — 45, 0, 5, 55, 5, 1,
35, 3, 1, 5, 16, 7 — are the seed-averaged version of the fourth column, so their
mean is the merged Generalization score, 14.83.

## 3. Where 25.56 comes from

Gen-Std is the standard half on its own. Per-seed means are 25.67 / 27.33 / 23.67,
giving **25.56 ± 1.50**. The 12 standard-half task SRs are:

```
stack_bowls 88, push_T 0, pack_objects_into_box 9, fold_clothes 87,
hang_mugs 11, sweep_blocks 3, pour_liquid_into_cup 55, make_toast 4,
arrange_largest_number 1, sort_nesting_dolls_by_size 9,
store_laptop_and_headphones 25, stack_blocks 15
```

Gen-Rand is the random half: per-seed 3.33 / 3.67 / 5.33 → **4.11 ± 0.87**.

The two exist precisely to separate in-distribution layouts from randomized ones,
which is the comparison in §5.3.2 and Figure 16b.

## 4. How Avg is computed

Not a flat mean over tasks, and not a six-way mean that treats Gen-Std and
Gen-Rand as separate dimensions. From `summarize_result.py::overall_seed_value`:

> Return (sr, score) at one seed as the mean of the per-dimension values.

There are five dimensions, and Generalization enters with its merged value:

```
(14.83 + 9.25 + 25.33 + 9.11 + 1.08) / 5 = 11.92
```

The same aggregation applied to Xiaomi-Robotics-1 gives its published 13.93:

```
((28.00 + 6.00)/2 + 18.83 + 23.67 + 6.56 + 3.58) / 5 = 13.93
```

A flat mean over the 42 tasks would give 12.33, not 11.92, because the dimensions
hold 12 / 8 / 8 / 6 / 8 tasks — a dimension mean is not a task mean.

## 5. The real defect: the bundle's per-task field is not consistent across models

The issue is right that OpenWAM-α does not line up with the rest of the
leaderboard. Running `verify_robodojo_scores.py --leaderboard` against the live
bundle, for all 37 entries:

| What the per-task field holds | Models |
|---|---|
| the **standard half** | **31** |
| the merged value | 0 |
| neither reading matches the published Gen-Std | **1 — OpenWAM-α** |

So the exported per-task field carries the standard half for 31 models and the
merged value for OpenWAM-α. Reading OpenWAM-α's row the way every other row must
be read is what produces 14.83 against a published 25.56. **That inference was
reasonable and the mismatch is real; it is an export inconsistency, not a scoring
one.**

A cross-check that needs none of our data: Xiaomi-Robotics-1 has
`stack_bowls` = 87 and `stack_bowls_random` = 16. If 87 were a merged value, the
standard half would have to be `2 × 87 − 16 = 158%`. So for that model the field
is unambiguously the standard half.

Two things follow, and both matter:

- **Every published aggregate we could check is correct**, for all 32 models with
  complete per-task data, OpenWAM-α included. The inconsistency is confined to
  which column the per-task API exposes.
- **Figure 16b compares like with like.** Every value plotted there is a
  standard-half number: OpenWAM-α 25.56, Xiaomi-Robotics-1 28.00,
  GalaxeaVLA (G0.5) 20.00, DM0.5 18.00.

We have reported the export inconsistency to the RoboDojo maintainers. Until it
is fixed, anyone auditing the leaderboard the same way will reach the same wrong
conclusion, which is why this directory exists.

## 6. Why the gap looked anomalous

Under the reading in the issue, the apparent discrepancy for any model is forced
to be exactly half its standard-to-random gap:

```
Gen-Std − merged = Gen-Std − (Gen-Std + Gen-Rand)/2 = (Gen-Std − Gen-Rand)/2
```

For OpenWAM-α that is `(25.56 − 4.11)/2 = 10.72`, matching the observed
`25.56 − 14.83 = 10.73`. The policy with the widest in-distribution /
out-of-distribution gap therefore shows the widest apparent discrepancy. Given
OpenWAM-α has the largest such gap on the board (−21.44), it surfaces first —
a property of the reading, not evidence about the scores.

## Raw data

`openwam_robodojo_per_seed.json` carries the full per-seed table this document is
built from. The underlying `_result.json` files for every (task, seed) are
available on request, and we are happy to walk through any individual row.
