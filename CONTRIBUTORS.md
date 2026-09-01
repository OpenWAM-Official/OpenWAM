# Contributors

OpenWAM retains a sanitized, per-commit history so that public work remains attributable to the people who contributed it. Contributors whose retained work consists only of excluded private material are not listed. Automation and release-maintenance commits are also excluded.

## Workload snapshot

The table is calculated from the sanitized history. **Authored commits** counts non-merge commits. **Co-authored commits** counts reviewed human `Co-authored-by` trailers on retained public commits. **Author changed lines** excludes vendored, generated, binary, and data-artifact paths. **Shared-commit lines** applies the same exclusions to the complete commits carrying that person's verified co-author trailer. **Surviving lines** uses `git blame -w -M -C -C` at the snapshot revision.

Git does not record which lines inside a shared commit were written by each co-author. Shared-commit lines therefore show the auditable scope of joint work, not personal line ownership; they can overlap author totals and are not added into repository totals.

Snapshot before this report: `7e47df93a422`.

| GitHub | Authored commits | Co-authored commits | Author changed lines | Shared-commit lines | Surviving lines | Main areas |
|---|---:|---:|---:|---:|---:|---|
| [@wayrise](https://github.com/wayrise) | 278 | 4 | 409,357 | 9,048 | 56,167 | model, tests, dataloader |
| [@KraHsu](https://github.com/KraHsu) | 77 | 3 | 59,478 | 7,750 | 19,882 | tests, model, benchmarks |
| [@d-finite](https://github.com/d-finite) | 68 | 0 | 36,054 | 0 | 20,166 | tests, model, dataloader |
| [@knightnemo](https://github.com/knightnemo) | 63 | 0 | 170,004 | 0 | 9,432 | other, docs, model |
| [@LeopoldYao](https://github.com/LeopoldYao) | 44 | 0 | 12,214 | 0 | 2,080 | benchmarks, tests, dataloader |
| [@SCreatorX](https://github.com/SCreatorX) | 39 | 0 | 12,752 | 0 | 5,871 | benchmarks, dataloader, tests |
| [@WayneJin0918](https://github.com/WayneJin0918) | 19 | 0 | 10,307 | 0 | 5,239 | dataloader, benchmarks, tests |
| [@xueminchi](https://github.com/xueminchi) | 9 | 0 | 7,141 | 0 | 2,358 | tests, model, core |
| [@yuechen0614](https://github.com/yuechen0614) | 8 | 0 | 13,232 | 0 | 9,719 | benchmarks, tests, dataloader |
| [@stubborn111](https://github.com/stubborn111) | 5 | 0 | 1,402 | 0 | 919 | dataloader, tests |
| [@Skywalker-yqz](https://github.com/Skywalker-yqz) | 5 | 0 | 929 | 0 | 114 | core, model, tests |
| [@Imnondeersty](https://github.com/Imnondeersty) | 4 | 0 | 2,956 | 0 | 2,639 | benchmarks, tests, dataloader |
| [@GuanqiHe](https://github.com/GuanqiHe) | 4 | 0 | 584 | 0 | 354 | tests, training, docs |
| [@guankou](https://github.com/guankou) | 3 | 0 | 7,135 | 0 | 91 | tests, dataloader, build CI tooling |
| [@Correr-Zhou](https://github.com/Correr-Zhou) | 2 | 0 | 1,376 | 0 | 1,082 | tests, model, dataloader |
| [@zhucoffee](https://github.com/zhucoffee) | 2 | 0 | 2,085 | 0 | 1,077 | model, tests, training |
| [@zhangwt20011015](https://github.com/zhangwt20011015) | 2 | 0 | 1,998 | 0 | 212 | dataloader, core, model |
| [@ewykric](https://github.com/ewykric) | 1 | 0 | 52 | 0 | 50 | model |
| [@Q-M-D](https://github.com/Q-M-D) | 1 | 0 | 23 | 0 | 5 | build CI tooling, training |

Totals at the snapshot: 19 human contributors, 634 human-authored non-merge commits, and 7 verified human co-author occurrences.

These figures are an auditable history snapshot, not a ranking of impact. Reviews, design work, debugging, mentoring, and coordination are not fully represented by Git metadata.
