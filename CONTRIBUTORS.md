# Contributors

OpenWAM retains a sanitized, per-commit history so that public work remains attributable to the people who contributed it. Contributors whose retained work consists only of excluded private material are not listed. Automation and release-maintenance commits are also excluded.

## Workload snapshot

The table is calculated from the sanitized history. **Authored commits** counts non-merge commits. **Co-authored commits** counts reviewed human `Co-authored-by` trailers on retained public commits. **Changed lines** excludes vendored, generated, binary, and data-artifact paths. **Surviving lines** uses `git blame -w -M -C -C` at the snapshot revision.

Git does not record which lines inside a shared commit were written by each co-author, so changed-line and blame metrics are credited only to the commit author; co-author work is represented by its independently verified commit count.

Snapshot before this report: `db308080991c`.

| GitHub | Authored commits | Co-authored commits | Changed lines | Surviving lines | Main areas |
|---|---:|---:|---:|---:|---|
| [@wayrise](https://github.com/wayrise) | 279 | 4 | 414,942 | 55,630 | model, tests, dataloader |
| [@KraHsu](https://github.com/KraHsu) | 76 | 3 | 59,657 | 19,943 | tests, model, benchmarks |
| [@d-finite](https://github.com/d-finite) | 67 | 0 | 34,903 | 17,989 | tests, model, dataloader |
| [@knightnemo](https://github.com/knightnemo) | 63 | 0 | 170,010 | 9,441 | other, docs, model |
| [@LeopoldYao](https://github.com/LeopoldYao) | 44 | 0 | 12,214 | 2,071 | benchmarks, tests, dataloader |
| [@SCreatorX](https://github.com/SCreatorX) | 37 | 0 | 12,716 | 5,674 | benchmarks, tests, dataloader |
| [@WayneJin0918](https://github.com/WayneJin0918) | 19 | 0 | 10,360 | 5,145 | dataloader, benchmarks, tests |
| [@xueminchi](https://github.com/xueminchi) | 9 | 0 | 7,130 | 2,354 | tests, model, core |
| [@yuechen0614](https://github.com/yuechen0614) | 6 | 0 | 13,205 | 9,697 | benchmarks, tests, dataloader |
| [@GuanqiHe](https://github.com/GuanqiHe) | 5 | 0 | 709 | 371 | training, tests, docs |
| [@Skywalker-yqz](https://github.com/Skywalker-yqz) | 5 | 0 | 929 | 114 | core, model, tests |
| [@stubborn111](https://github.com/stubborn111) | 4 | 0 | 1,144 | 546 | dataloader, tests |
| [@Imnondeersty](https://github.com/Imnondeersty) | 3 | 0 | 2,900 | 2,582 | benchmarks, tests, dataloader |
| [@guankou](https://github.com/guankou) | 3 | 0 | 7,150 | 89 | tests, dataloader, build CI tooling |
| [@Correr-Zhou](https://github.com/Correr-Zhou) | 2 | 0 | 1,379 | 1,083 | tests, model, dataloader |
| [@zhucoffee](https://github.com/zhucoffee) | 2 | 0 | 2,085 | 1,077 | model, tests, training |
| [@zhangwt20011015](https://github.com/zhangwt20011015) | 1 | 0 | 2,744 | 1,026 | dataloader, training, core |
| [@ewykric](https://github.com/ewykric) | 1 | 0 | 52 | 50 | model |
| [@Q-M-D](https://github.com/Q-M-D) | 1 | 0 | 294 | 5 | tests, training, build CI tooling |

Totals at the snapshot: 19 human contributors, 627 human-authored non-merge commits, and 7 verified human co-author occurrences.

These figures are an auditable history snapshot, not a ranking of impact. Reviews, design work, debugging, mentoring, and coordination are not fully represented by Git metadata.
