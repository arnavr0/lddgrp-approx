# LDdGRP ALNS experimental report

## Scope and correctness

The solver was evaluated on all 84 official LDdGRP instances published by the
authors. The parser reads the required-edge service times and coordinates from
each `.dat` file, reconstructs its complete Euclidean deadheading metric, and
takes instance-specific curb weight `w_0` and landing/takeoff cost `c_D` from
the accompanying result table.

The method is a feasible stage-aware ALNS, **not** an approximation algorithm
and **not** an exact LDdGRP solver. Its candidates are independently decoded
into complete route certificates; validation checks route continuity, every
required-edge traversal, delivery-load multipliers, and the displayed
objective. It uses exact Held--Karp decoding for individual stages with at
most ten required edges and a 12-pass orientation-aware 2-opt/relocation
decoder for larger stages.

Official sources: [instance page](https://www.uv.es/plani/instancias.html),
[instance archive](https://www.uv.es/plani/data/LDdGRP/LDdGRP%20instances.rar),
and [published parameter/bound table](https://www.uv.es/plani/data/LDdGRP/LDdGRP%20results.txt).

## Reproducible 1-second baseline

Configuration: one ALNS start, seed 7, at most 2,000 outer iterations, and a
one-second per-instance limit. Construction receives 25% of that limit. The
full certificates and all values are in `official-vnd12-1s.json`.

| Measure | Result |
|---|---:|
| Instances | 84 |
| Mean difference from published upper bound | 17.535% |
| Median difference from published upper bound | 17.532% |
| Lowest difference | 2.745% |
| Highest difference | 34.783% |
| Mean recorded elapsed time | 1.047 s |
| Published upper bounds labelled optimal | 66 |
| Published upper bounds not proved optimal | 18 |

The comparison is `100 * (heuristic objective - published upper bound) /
published upper bound`. For the 66 rows marked `optimo`, it is a gap to an
optimum. For the other 18, it is only a difference from the authors' reported
incumbent; it is not an optimality gap.

| Grid size | Mean difference from published upper bound |
|---|---:|
| 4 x 4 | 24.538% |
| 5 x 5 | 17.747% |
| 6 x 6 | 19.586% |
| 7 x 7 | 16.534% |
| 8 x 8 | 15.760% |
| 9 x 9 | 15.330% |
| 10 x 10 | 13.247% |

## Controlled ablations

All values below use the 12 10 x 10 instances, seed 7, one start, and the same
2,000-iteration cap unless otherwise noted. They establish a direction for
further work, not a statistical claim across random seeds.

| Change | Budget | Mean difference from published upper bound |
|---|---:|---:|
| Greedy large-stage decoder (no descent) | 1 s | 27.985% |
| Six VND passes | 1 s | 14.988% |
| Nine VND passes | 1 s | 13.677% |
| Twelve VND passes (baseline) | 1 s | 13.260% |
| Exact fixed-edge-stage delivery-order DP | 1 s | 14.151% |
| Frozen-skeleton two-stage exact repair | 1 s | 13.552% |
| Twelve VND passes (baseline), seed 7 | 5 s | 12.171% |
| Exact per-stage decoding through 12 edges, seed 7 | 5 s | 13.743% |
| Frozen-skeleton two-stage exact repair, seed 7 | 5 s | 11.025% |
| Twelve VND passes (baseline), seed 19 | 5 s | 11.010% |
| Frozen-skeleton two-stage exact repair, seed 19 | 5 s | 10.230% |

The routing VND is therefore the largest verified gain and is the default. The
delivery-order DP is mathematically exact for fixed stage labels, but its
high-fidelity route evaluations consume too much of a one-second search. The
frozen-skeleton repair is also not suitable for the short default budget. At
five seconds it improved the same 10 x 10 group for two independent seeds. The
paired pooled mean fell from 11.591% to 10.628% (a 0.963 percentage-point
reduction; 14 of 24 paired instance-runs improved). This supports using it as
an opt-in long-budget intensification, but it remains unconfirmed on the full
84-instance suite.

Increasing the exact per-stage Held--Karp threshold from 10 to 12 was also
counterproductive at five seconds (13.743% for seed 7): the extra exponential
decoding displaced more useful outer-search work. The default threshold stays
at 10.

The earlier spatial two-stage repair was also retained only as an ablation: on
the full 84-instance one-second suite it had a 17.638% mean comparison versus
17.535% for the default.

## Interpretation

This is a substantial improvement over the initial 25.690% one-second
prototype, but it is still not competitive with the authors' exact results.
The authors' branch-and-cut used a different C++/CPLEX implementation,
hardware, and a two-hour cap; because the solution qualities and environments
differ, reporting a wall-clock “speedup” would be invalid.

The credible next step is not an untested end-to-end ML or genetic algorithm.
It is a time-limited exact LDdGRP local-repair model that fixes the outside
solution and optimizes a destroyed neighbourhood with the original model's
connectivity constraints. The current `EdgeStageRepairOracle` interface is
intended to accept that backend. Learning can then select repair size, destroy
operator, and time allocation after it has a strong matheuristic baseline to
improve.
