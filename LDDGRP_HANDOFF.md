# LDdGRP research and implementation handoff

**Last updated:** 2026-10-21  
**Purpose:** Read this file first when continuing this work in a new chat. It
records what is established, what is implemented, what was measured, and what
must not be claimed.

## Objective and current conclusion

The project investigates practical approximate methods for the discrete
**load-dependent drone general routing problem (LDdGRP)** and implements a
correctness-first ALNS/matheuristic prototype.

The current solver returns valid feasible solutions quickly, but is **not yet
competitive with the authors' exact branch-and-cut solutions**. The best short
budget configuration is the default 12-pass large-stage decoder. A frozen-route
skeleton repair is promising only with a longer budget and remains opt-in.

Do not claim:

- an approximation guarantee or global optimality;
- a same-hardware speedup over the authors' exact method;
- that a comparison against a non-optimal published upper bound is an
  optimality gap.

## Primary sources

- Exact LDdGRP formulation and branch-and-cut: Paula Segura Martínez,
  *Arc routing problems with drones*, Chapter 7 (2023),
  <https://webges.uv.es/public/uvEntreuWeb/tesis/tesis-2723774-CO6D30E9IUIIIGUP.pdf>
- Official instance page: <https://www.uv.es/plani/instancias.html>
- Official 84-instance archive:
  <https://www.uv.es/plani/data/LDdGRP/LDdGRP%20instances.rar>
- Official parameter/bound table:
  <https://www.uv.es/plani/data/LDdGRP/LDdGRP%20results.txt>

Relevant facts from Chapter 7:

- The authors' implementation is C++ plus single-threaded CPLEX 20.1 on a
  3.4 GHz Intel Core i7 with 16 GB RAM; CPLEX built-in heuristics and cuts were
  disabled.
- They use a two-hour limit. They prove 66 of 84 benchmark instances optimal;
  the remaining 18 published upper bounds are incumbents, not optima.
- Their average B&C times by grid size (4x4 through 10x10) are respectively
  8.2, 206.0, 895.1, 1756.3, 3519.3, 3161.8, and 4332.3 seconds.
- There is no known direct published ALNS/ML/RL/GA method specifically for this
  discrete LDdGRP formulation in the research completed for this project.
  Related load-dependent arc-routing work is not automatically a valid LDdGRP
  benchmark or method comparison.
- **Important disambiguation:** the thesis does contain a VND matheuristic,
  but it is Chapter 6's method for the *multi-purpose K-drones GRP* (MP
  K-DGRP), a multi-drone problem with duration and capacity constraints. It is
  not an LDdGRP GRASP/VND method. Chapter 7, which defines LDdGRP, contains
  the formulation, polyhedral analysis, and branch-and-cut only. The thesis
  has no searchable occurrence of “GRASP”; the 2024 LDdGRP SSRN abstract also
  describes a formulation, valid inequalities, branch-and-cut, and
  computational tests—not a heuristic.

## Problem model used here

The implementation represents the discrete model from Chapter 7.

- One drone starts at a depot, visits every delivery node exactly once, services
  every required edge exactly once, and returns to the depot.
- A delivery sequence creates `d + 1` stages, where `d` is the number of
  deliveries.
- Every required edge is assigned to exactly one stage.
- A stage's base route cost consists of deadheading plus required-edge service
  costs. Its weighted cost is
  `(curb_weight + remaining delivery load) * (base_cost + c_D)`.
- Deadheading is a complete symmetric triangle-metric cost matrix. Official
  benchmarks reconstruct it from Euclidean coordinates.
- The generic routing-format `coste` attached to delivery vertices in the data
  file is **not** silently treated as `c_D`; `w_0` and `c_D` come from the
  official results table.

## Repository layout

The project root is:

`C:\Users\Arnav Rohilla\Documents\Codex\2026-10-08\what-recent-work-has-been-done`

Important files:

| Path | Role |
|---|---|
| `src/lddgrp/model.py` | Validated LDdGRP data model. |
| `src/lddgrp/solver.py` | Candidate representation, decoder, ALNS, restricted exact repairs. |
| `src/lddgrp/repair.py` | `EdgeStageRepairOracle` protocol and bounded enumerative repair. |
| `src/lddgrp/benchmarks.py` | Official-data parser, reproducible runner, checkpoint/resume protection. |
| `src/lddgrp/cli.py` | Single-instance JSON input/output with route certificate. |
| `tests/test_solver.py` | Solver, decoder, repair, and certificate tests. |
| `tests/test_benchmarks.py` | Official-parser test. |
| `README.md` | User-facing usage documentation. |
| `outputs/lddgrp-benchmark-report.md` | Detailed experimental report. |

This directory is **not a Git repository**. Do not use `git diff` or `git
status` as verification.

## Implemented solver

### Candidate and certificate

`Candidate` contains:

1. `delivery_order`: a permutation of delivery nodes.
2. `edge_stages`: one zero-based stage label per required edge.

`evaluate()` decodes every stage and returns `Evaluation`, including a route
certificate composed of contiguous deadhead and service steps. Every result is
checked by `validate_evaluation()` for service coverage, path continuity,
stage endpoints, load multiplier, and exact objective recomputation.

### Default short-budget configuration

`ALNSConfig` defaults used in the published one-second experiment:

- 1.0 second per instance, 2,000 max outer iterations;
- 25% construction allocation;
- exact per-stage Held--Karp threshold: 10 required edges;
- 12 orientation-aware large-stage VND passes;
- local enumerative repair: at most 7 mutable edges;
- spatial repair, stage descent, delivery-order DP, and skeleton repair:
  **disabled**.

The one-second strategy is:

1. Build a demand-aware delivery order.
2. Assign required edges by true decoded marginal cost across stages.
3. Decode a small stage exactly via Held--Karp; decode a larger stage with
   nearest-service construction followed by orientation-aware 2-opt and
   relocation VND.
4. Run adaptive large-neighbourhood search with simulated-annealing acceptance:
   delivery relocate/swap/reverse/ruin-recreate; single-edge and cluster stage
   moves; cluster ruin/recreate; and bounded two-stage enumerative repair.
5. Re-decode and validate the final best candidate.

Stage-route cache keys include decoder fidelity, preventing a low-fidelity
route from being reused for final certification.

### Restricted exact components

These are exact only in their explicitly fixed neighbourhoods. None proves a
global LDdGRP optimum.

- **Small-stage Held--Karp:** exact sequence/orientation for stages with at
  most `exact_stage_limit` edges.
- **Enumerative edge-stage repair:** exact binary assignment of a small selected
  edge set across two stages; currently at most 7 mutable edges.
- **Delivery-order DP:** exact delivery sequence given all edge-stage labels.
  Disabled by default because its route evaluations consumed too much of a
  one-second budget.
- **Frozen-skeleton two-stage repair:** fix the incumbent service order and
  direction of all non-destroyed edges in two chosen stages. For every subset
  of up to seven mutable edges, dynamic programming finds the best insertion
  order/orientations around the fixed skeleton. It then chooses the best binary
  split of the cluster between the two stages. The standard full decoder must
  confirm a strict improvement before the candidate is retained.
- **Load-aware ruin/recreate:** remove up to six service edges with the largest
  current route-removal saving multiplied by the stage load multiplier, then
  reinsert them by weighted regret. It is a heuristic, not an exact
  subproblem; it is run only in a bounded final window when enabled.
- **Decoded-route block relocation:** choose one costly contiguous service
  block (at most four required edges) from the decoded certificate and test
  moving its edge labels to every other stage. This is a cheap arc-routing
  inspired neighbourhood, not an exact route-block move because the ordinary
  decoder is free to reorder the target stage.
- **Contiguous sequence/split VNS:** flatten the decoded service order and,
  for a fixed delivery order, exactly choose directions and contiguous stage
  boundaries with dynamic programming. It can then test bounded
  swap/relocate/reverse/block perturbations of that sequence. The contiguity
  restriction is material, so it is not a global LDdGRP exact method.
- **Giant-tour VND:** represent every solution as one permutation of delivery
  tasks and required-edge service tasks. A two-state orientation DP scores the
  fixed task order exactly; delivery positions define both the delivery order
  and edge-stage allocation. The current VND scans relocate and short Or-opt
  (blocks of one or two tasks), then projects a proposed tour into a complete
  stage certificate. Each stage keeps the cheaper of its giant-tour route and
  ordinary decoder route before `validate_evaluation()` can accept it. Thus it
  is exact only for the supplied giant-tour order and orientations, not a
  global LDdGRP exact method.

Enable the exact skeleton repair deliberately:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
py -3.13 -m lddgrp.benchmarks work/lddgrp-benchmarks work/LDdGRP-results.txt `
  --pattern 'LDdGRP10_10_*.dat' --seconds 5 --iterations 10000 --seed 7 `
  --large-stage-local-search-passes 12 --enable-skeleton-repair `
  --output outputs/experiment.json
```

The sequence/split long-budget ablation uses the following full configuration;
the runner writes every parameter to JSON and rejects incompatible `--resume`
requests:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
py -3.13 -m lddgrp.benchmarks work/lddgrp-benchmarks work/LDdGRP-results.txt `
  --pattern 'LDdGRP10_10_*.dat' --seconds 5 --iterations 10000 --seed 7 `
  --large-stage-local-search-passes 12 --enable-sequence-split-vnd `
  --sequence-split-seconds 0.10 --sequence-split-max-moves 12 `
  --sequence-split-final-seconds 0.12 `
  --output outputs/official-grid10-sequence-split-final-fixed-5s.json
```

The giant-tour three-second ablation is reproducible as follows. Its final
window is a cap; the implementation scales the actual reservation down on
smaller instances because the scanned neighbourhood is quadratic in task count.

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
py -3.13 -m lddgrp.benchmarks work/lddgrp-benchmarks work/LDdGRP-results.txt `
  --pattern 'LDdGRP10_10_*.dat' --seconds 3 --iterations 6000 --seed 7 `
  --large-stage-local-search-passes 12 --enable-giant-tour-vnd `
  --giant-tour-final-seconds 0.80 --giant-tour-max-passes 2 `
  --giant-tour-max-block-size 2 `
  --output outputs/official-grid10-giant-tour-vnd-3s-seed7.json
```

## Complexity and speed claims

Let `m` be required edges, `d` delivery nodes, `q` the size of one stage,
`B=10` the exact-stage cap, `P=12` VND passes, and `R=7` the repair cap.

- Small-stage decoder: `O(2^q q^2)` time for `q <= B`.
- Large-stage decoder: approximately `O(P q^2)` time.
- Ignoring cache hits and treating the configured caps as constants, one full
  evaluation is approximately `O(P m^2 + (d+1) 2^B B^2)`.
- Bounded binary repair has at most `2^R = 128` allocations.
- The initializer is more expensive than one evaluation because it tests all
  stages while assigning edges; a coarse upper bound is `O((d+1) P m^3)`.

The authors' branch-and-cut has exponential worst-case behavior. That does
**not** establish a useful asymptotic or practical speedup: their method
optimizes/bounds globally, while this one has a quality trade-off and a hard
wall-clock budget.

Measured short-budget timing on the current host, not the authors' hardware:

- 84-instance mean 1.046874 s; median 1.034012 s; min 1.000372 s; max
  1.196479 s.

Dividing published B&C times by those current-host times produces an apparent
ratio from 8x (4x4) to 4,224x (10x10), 1,894x overall. **Never label those
values speedups.** They use different machines, implementation languages,
solver stacks, stopping criteria, and solution quality. A valid same-hardware
comparison requires the authors' executable/model and CPLEX settings to be run
alongside this code with a fixed equal-quality or equal-time protocol.

## Experimental evidence

All gap-like figures below are differences from the authors' published upper
bound:

`100 * (heuristic objective - published upper bound) / published upper bound`.

For only the 66 `optimo` rows, this is a gap to a proven optimum. For the other
18 it is merely a difference from an incumbent.

### Main 84-instance one-second baseline

Artifact: `outputs/official-vnd12-1s.json`

| Measure | Value |
|---|---:|
| Mean | 17.535% |
| Median | 17.532% |
| Minimum | 2.745% |
| Maximum | 34.783% |
| Mean time | 1.047 s |

By grid size, means are: 4x4 24.538%, 5x5 17.747%, 6x6 19.586%,
7x7 16.534%, 8x8 15.760%, 9x9 15.330%, and 10x10 13.247%.

This substantially improves the earliest 25.690% prototype but is not
competitive with the exact B&C outcomes.

### Controlled 10x10 ablations

All one-second/5-second comparisons use the 12 official 10x10 instances, one
start, 2,000 or 10,000 iterations respectively, and the stated seed.

| Method | Budget | Result |
|---|---:|---:|
| Greedy large-stage decoder (0 VND passes) | 1 s, seed 7 | 27.985% |
| 6 VND passes | 1 s, seed 7 | 14.988% |
| 9 VND passes | 1 s, seed 7 | 13.677% |
| 12 VND passes | 1 s, seed 7 | 13.260% |
| Delivery-order DP enabled | 1 s, seed 7 | 14.151% |
| Spatial two-stage repair | 1 s, seed 7 | 13.088% |
| Frozen-skeleton repair | 1 s, seed 7 | 13.552% |
| Sequence/split VNS, 0.12 s final window | 1 s, seed 7 | 12.681% |
| Sequence/split VNS, 0.12 s final window | 1 s, seed 19 | 13.129% |
| Sequence split only, 0.04 s final window | 1 s, seeds 7/19 pooled | 13.339% |
| Baseline | 5 s, seed 7 | 12.171% |
| Frozen-skeleton repair | 5 s, seed 7 | 11.025% |
| Baseline | 5 s, seed 19 | 11.010% |
| Frozen-skeleton repair | 5 s, seed 19 | 10.230% |
| Sequence/split VNS, 0.12 s final window | 5 s, seed 7 | 10.763% |
| Sequence/split VNS, 0.12 s final window | 5 s, seed 19 | 11.078% |
| Exact-stage limit raised to 12 | 5 s, seed 7 | 13.743% |

Interpretation:

- The 12-pass large-stage VND is the largest confirmed short-budget gain.
- Delivery-order DP and frozen-skeleton repair are counterproductive at one
  second because they displace too much outer search.
- The spatial repair improved this particular 10x10 run but worsened the
  full 84-instance one-second mean: 17.638% versus 17.535%. Keep it disabled.
- Over the two paired five-second skeleton tests, the pooled mean falls from
  11.591% to 10.628%, an improvement of 0.963 percentage points; 14 of 24
  paired instance-runs improve. This is promising but **not yet a full-suite,
  multi-seed result**.
- Increasing the Held--Karp threshold to 12 consumes too much time and worsens
  the result. Leave the threshold at 10.
- The load-aware ruin/recreate and decoded route-block operators were both
  rejected at one second. The former was neutral-to-worse over three seeds;
  the latter increased the seed-7 10x10 mean from 13.260% to 13.670%.
- The sequence/split results in files without `-fixed-` are **invalid
  experimental evidence**. A deadline bug caused the post-search split repair
  to see the already-expired ALNS cutoff, so those runs measured a reserved
  time loss rather than the intended repair. The bug is covered by a regression
  test; use only the corrected `-fixed-` artifacts below.
- With the corrected implementation, the ordinary sequence/split VNS is not
  a one-second improvement: across matched seeds 7 and 19 it has mean paired
  objective change **+0.178%** (7 wins / 7 ties / 10 losses), and the mean
  reference difference rises from 12.731% to 12.905%. The lean 0.04-second,
  split-only version is worse still: +0.544% paired and 13.339% reference
  difference. Do not enable either in a short-budget preset.
- At five seconds, the corrected sequence/split VNS is encouraging but not
  conclusive. Against paired baselines it changes the mean objective by
  -1.273% for seed 7 (6/4/2 win/tie/loss) and +0.059% for seed 19 (4/1/7).
  Pooled: **-0.607%**, median 0%, **10/5/9** win/tie/loss; its pooled mean
  reference difference is 10.920% versus 11.591% for the two baselines. This
  is insufficient to change defaults or claim generalisation.

### Giant-tour VND: corrected three-second evidence

This is a distinct experiment from the one- and five-second ablations above.
Each run uses 6,000 iterations, the 12-pass stage decoder, one start, and the
same seed in its no-giant-tour baseline and giant-tour trial. The intensifier
uses a 0.80-second cap, two VND passes, and task blocks of at most two. All
reported changes below are paired objective changes, not cross-machine timing
claims.

| Group | Seeds | Baseline mean reference difference | Giant-tour mean reference difference | Paired objective change | Win / tie / loss |
|---|---|---:|---:|---:|---:|
| 10x10, 12 instances | 7 | 12.820% | 10.019% | -2.426% | 12 / 0 / 0 |
| 10x10, 12 instances | 19 | 11.593% | 9.270% | -2.049% | 12 / 0 / 0 |
| 10x10, 12 instances | 31 | 11.719% | 9.161% | -2.235% | 12 / 0 / 0 |
| 10x10 pooled, 36 runs | 7, 19, 31 | 12.044% | 9.483% | **-2.237%** (median -1.950%) | **36 / 0 / 0** |
| 7x7, 12 instances | 7 | 13.843% | 12.644% | -1.043% | 9 / 3 / 0 |
| 4x4, 12 instances | 7 | 23.640% | 23.651% | +0.009% | 0 / 11 / 1 |

Interpretation:

- The 10x10 finding is the first repeated, substantial improvement over
  VND-12 in the targeted 2.5--4.0 second regime. It directly supports the
  representation-coupling hypothesis for large instances.
- It is **not** yet an all-84 result, so it does not establish an overall
  default or replace the one-second VND-12 recommendation.
- The 4x4 result is an important negative control: the giant-tour relocate/
  Or-opt neighbourhood alone does not address its remaining delivery-order or
  stage-charge difficulty. The final-window reservation now scales with task
  count, but a time-limited stochastic baseline can still take a different
  search path after even a small budget change.
- The implementation was checked against all 84 stored VND-12 certificates:
  lift-to-giant-tour then fixed-order re-evaluation reproduced every objective
  to tolerance. Unit tests also exhaustively compare the incremental
  relocate/Or-opt scan with brute-force scoring on random small tours.

Key artifacts:

| Artifact | Meaning |
|---|---|
| `outputs/official-alns-cluster-1s.json` | Earlier 25.690% baseline. |
| `outputs/official-vnd12-1s.json` | Current recommended one-second baseline. |
| `outputs/official-grid10-passes0-1s.json` | VND 0-pass ablation. |
| `outputs/official-grid10-passes6-1s.json` | VND 6-pass ablation. |
| `outputs/official-grid10-passes9-1s.json` | VND 9-pass ablation. |
| `outputs/official-grid10-passes12-1s.json` | VND 12-pass ablation. |
| `outputs/official-grid10-dp-vnd12-1s.json` | Delivery-order DP ablation. |
| `outputs/official-grid10-spatial-repair-1s.json` | Spatial repair ablation. |
| `outputs/official-spatial-vnd12-1s.json` | Spatial repair, all 84 instances. |
| `outputs/official-grid10-vnd12-5s.json` | Baseline 5s, seed 7. |
| `outputs/official-grid10-skeleton-repair-5s.json` | Skeleton 5s, seed 7. |
| `outputs/official-grid10-vnd12-5s-seed19.json` | Baseline 5s, seed 19. |
| `outputs/official-grid10-skeleton-repair-5s-seed19.json` | Skeleton 5s, seed 19. |
| `outputs/official-grid10-exact12-vnd12-5s.json` | Rejected 12-edge exact-decoder ablation. |
| `outputs/official-grid10-sequence-split-final-fixed-1s.json` | Corrected sequence/split VNS, 1s, seed 7. |
| `outputs/official-grid10-sequence-split-final-fixed-1s-seed19.json` | Corrected sequence/split VNS, 1s, seed 19. |
| `outputs/official-grid10-sequence-split-only-final04-1s.json` | Corrected 0.04s split-only test, seed 7. |
| `outputs/official-grid10-sequence-split-only-final04-1s-seed19.json` | Corrected 0.04s split-only test, seed 19. |
| `outputs/official-grid10-sequence-split-final-fixed-5s.json` | Corrected sequence/split VNS, 5s, seed 7. |
| `outputs/official-grid10-sequence-split-final-fixed-5s-seed19.json` | Corrected sequence/split VNS, 5s, seed 19. |
| `outputs/official-grid10-vnd12-3s-seed7.json` | Matched 10x10 baseline, 3s, seed 7. |
| `outputs/official-grid10-giant-tour-vnd-3s-seed7.json` | Giant-tour VND, 3s, seed 7. |
| `outputs/official-grid10-vnd12-3s-seed19.json` | Matched 10x10 baseline, 3s, seed 19. |
| `outputs/official-grid10-giant-tour-vnd-3s-seed19.json` | Giant-tour VND, 3s, seed 19. |
| `outputs/official-grid10-vnd12-3s-seed31.json` | Matched 10x10 baseline, 3s, seed 31. |
| `outputs/official-grid10-giant-tour-vnd-3s-seed31.json` | Giant-tour VND, 3s, seed 31. |
| `outputs/official-grid7-vnd12-3s-seed7.json` | Matched 7x7 baseline, 3s, seed 7. |
| `outputs/official-grid7-giant-tour-vnd-3s-seed7.json` | Giant-tour VND, 3s, seed 7. |
| `outputs/official-grid4-vnd12-3s-seed7.json` | Matched 4x4 baseline, 3s, seed 7. |
| `outputs/official-grid4-giant-tour-vnd-adaptive-3s-seed7.json` | Giant-tour VND, 3s, seed 7. |

## Reproducibility and verification

Run tests from the repository root:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
py -3.13 -m pytest -q
```

Last checked result: **28 passed**.

The Python launcher may print a non-fatal message saying it cannot find the
real location of the Python 3.13 executable; tests and commands nevertheless
completed successfully in this environment.

Benchmark reports checkpoint after each instance. `--resume` now checks:

- time limit, iteration limit, seed, restarts;
- exact-stage limit and VND-pass count;
- skeleton-repair enable flag, mutable-edge count, and time cap;
- delivery-order-DP enable flag.
- load-aware ruin enable flag, destroy cap, per-call cap, final-window cap,
  and final-pass count;
- route-block relocation enable flag, block-size cap, and per-call cap;
- sequence/split VNS enable flag, per-call cap, move cap, and final-window
  cap.
- giant-tour VND enable flag, final-window cap, pass cap, and block-size cap.

Do not overwrite a benchmark report with an incompatible configuration. The
runner should reject it.

## Recommended continuation plan

### Highest-value next technical step

Complete the **all-84, matched three-second giant-tour protocol** before
changing a default: run at least seeds 7, 19, and 31 for both VND-12 and the
giant-tour configuration, report the `optimo` and incumbent-bound rows
separately, and inspect the grid-size interaction. The 10x10 three-seed result
is strong, but 4x4 is currently neutral.

After that validation, the next implementation priority is an exact local
repair backend behind `EdgeStageRepairOracle`:

1. Destroy a geographically coherent neighbourhood of stages/edges.
2. Fix all decisions outside it.
3. Optimize the inside with the original LDdGRP connectivity/parity structure
   or an equivalent exact route model.
4. Return only a feasible improved candidate and validate it through the
   existing certificate machinery.

This is the justified matheuristic direction. It improves the key weakness:
edge-stage assignment and cross-stage route sharing. It is more credible than
adding end-to-end RL, a genetic algorithm, or a nature-inspired method to the
current weak allocation search.

### Experimental plan before changing defaults

1. Run the giant-tour configuration and a clean VND-12 baseline on all 84
   instances at three seconds for seeds 7, 19, and 31. Resume reports only
   with exactly matching settings. Do not extrapolate 10x10 evidence to 4x4.
2. Report paired seed-level means, medians, win/tie/loss counts, and both
   reference differences and objective deltas. Treat the 18 non-`optimo`
   instances as comparisons to published incumbents, not proven optima.
3. If giant-tour VND generalizes, expose it as a documented 3-second preset;
   keep the one-second default unchanged unless a distinct fresh short-budget
   multi-seed experiment improves it.
4. Then run corrected sequence/split and skeleton configurations in their own
   five-second, complete multi-seed protocols. Do not combine operators until
   their individual effects are measured.
5. Then use a validated local-repair backend as a dataset generator for ML
   only if learned operator selection or neighbourhood sizing improves a
   held-out benchmark.

### If exploring a different approach

Good alternatives to test, in this order:

1. A large-neighbourhood exact MIP/CP-SAT/branch-and-cut repair preserving the
   original model constraints.
2. A route-pool/set-partitioning matheuristic using high-quality stage routes
   from ALNS as columns.
3. Adaptive operator/neighbourhood selection using simple state features;
   benchmark against the current fixed ALNS before invoking RL.
4. Memetic/GA methods only with this decoder and repair oracle as local search,
   not as raw edge-stage-string crossover.

Avoid claiming that generic CPP-LC, RPP-LC, or generic VRP papers directly
solve LDdGRP without checking model equivalence.

## Guardrails for future work

- Preserve any existing user edits; this directory may be dirty.
- Use `apply_patch` for source changes.
- Run tests after every solver change and retain only configurations supported
  by controlled comparisons.
- Keep results labelled by seed, time budget, iteration cap, and enabled
  operators.
- If browsing or making external claims, use the primary thesis and official
  benchmark sources above.
- Do not call an apparent published-time/current-time ratio a speedup.
- Do not say “100% certain”; explain verification coverage and remaining
  uncertainty instead.
