# LDdGRP ALNS prototype

This is a dependency-free, correctness-first prototype for the **discrete
Load-Dependent drone General Routing Problem (LDdGRP)**.  It implements the
most defensible approximate strategy for the presently available model:

- a delivery-order plus edge-stage solution representation;
- a coupled marginal-cost initializer that accounts for route sharing when it
  assigns sensing edges to load stages;
- an exact dynamic-programming decoder for small individual stages;
- an orientation-aware 2-opt and relocation decoder for larger stages;
- adaptive large-neighbourhood search (ALNS), with simulated-annealing
  acceptance, delivery/edge-cluster destroy-and-repair moves, and bounded
  exact enumeration for an affected two-stage edge-allocation neighbourhood;
- a multi-stage, spatial edge-cluster ruin/recreate move that scores decoded
  marginal stage costs during reconstruction;
- an exact, bounded spatial-cluster repair that repartitions up to six related
  edges between two delivery stages (available for ablation, disabled by
  default because it did not improve the full one-second benchmark);
- an exact frozen-route-skeleton repair: for two selected stages, it preserves
  every non-destroyed required-edge service order and direction, then uses a
  dynamic program to optimize the placement and orientation of every subset
  of up to seven destroyed edges (also an opt-in ablation);
- a parser and batch runner for the authors' 84 published LDdGRP instances;
- JSON input/output and full route certificates.

It is a heuristic, not an approximation algorithm: it does **not** provide an
optimality guarantee.  Every returned solution is nevertheless explicitly
decoded and validated against the discrete LDdGRP constraints.

## Model represented

The implementation follows the discrete model in Plana, Sanchis and Segura:
one drone starts and ends at a depot; it visits each delivery node exactly
once; it services every required edge; and the travel/service cost of a stage
is multiplied by the drone's curb weight plus its remaining delivery load.
The non-required network is represented by a complete, symmetric metric cost
matrix.

A solution has two coupled decisions:

1. the order of delivery nodes;
2. the stage (the interval between two consecutive deliveries) in which each
   required edge is serviced.

For a fixed stage, the decoder chooses an order and orientation for its
required edges.  With a complete metric deadheading graph and a parallel
non-required edge available for each required edge, a stage can be represented
as an endpoint-to-endpoint sequence of once-serviced required edges.  The
decoder solves this exactly by bitmask DP when the stage contains at most 10
required edges; above that threshold it uses a deterministic greedy route.

## Input format

`examples/tiny.json` is a complete example.  Node identifiers must be the
integer indices of `travel_cost`.  The cost matrix must be symmetric, have a
zero diagonal, and satisfy the triangle inequality (within numerical
tolerance).

```json
{
  "depot": 0,
  "travel_cost": [[0, 3], [3, 0]],
  "delivery_demands": {"1": 4.0},
  "required_edges": [{"u": 0, "v": 1, "service_cost": 5.0}],
  "curb_weight": 2.0,
  "landing_takeoff_cost": 0.25
}
```

`service_cost` is the base cost of traversing and servicing the required edge;
it must be at least the corresponding deadheading cost.

## Run

```powershell
py -3.13 -m pip install -e .
py -3.13 -m lddgrp.cli examples/tiny.json --seconds 5 --seed 7 --output outputs/solution.json
py -3.13 -m pytest -q
```

The JSON output includes the delivery order, edge-to-stage allocation, each
decoded stage route, and the objective breakdown.  It is deliberately suitable
as a warm-start producer for a future exact branch-and-cut repair component.

## Official benchmark data

The implementation reads the authors' original `.dat` files and the associated
`LDdGRP results.txt` table.  The table supplies the instance-specific curb
weight `w_0` and delivery landing/takeoff parameter `c_D`; those values are
not stored in the `.dat` file.  The generic routing-format `coste` column on
required vertices is retained in the benchmark report as metadata, but is not
substituted for `c_D`: Chapter 7's LDdGRP objective uses the latter.

```powershell
py -3.13 -m lddgrp.benchmarks work/lddgrp-benchmarks work/LDdGRP-results.txt `
  --pattern 'LDdGRP4_4_2_1.dat' --seconds 5 --restarts 4 --seed 7 `
  --output outputs/official-smoke.json
```

`gap_to_published_upper_bound_percent` is a comparison with the authors'
reported incumbent/optimum, not an optimality certificate for this heuristic.
The runner checkpoints after each completed instance.  If a long experiment is
interrupted, rerun the same command with `--resume`; it verifies the time,
iteration, and seed settings before reusing records.

Within each per-instance limit, the coupled construction receives 25% of the
time by default and ALNS receives the remainder.  This prevents a stronger
initializer from consuming an entire very short experiment budget.
`--restarts` divides that same per-instance budget across independent starts;
the report includes all restart objectives and the best route certificate.
For ablations, `--large-stage-local-search-passes` controls the number of
orientation-aware 2-opt/relocation passes in stages above the exact-DP limit
(the measured default is 12).
The default uses the same 12-pass decoder during search and final
certification.  A two-tier mode can be configured in Python for ablation; its
cache keys include decoder fidelity, so a lower-fidelity route is never reused
as a final result.  It did not improve the one-second 10 x 10 experiment.

For a fixed assignment of required edges to stages, the solver also optimizes
the entire delivery sequence exactly with a subset dynamic program.  This is a
restricted exact subproblem, not a global LDdGRP optimality claim.  It is
available as the `--enable-delivery-order-dp` ablation rather than enabled by
default: on the controlled one-second 10 x 10 experiment, its extra decoding
time increased the mean comparison with the published upper bounds.

`--enable-skeleton-repair` enables a different restricted exact repair.  It
does not freeze the destroyed edges: it evaluates every binary allocation of a
small cluster across two stages, while a dynamic program finds its optimal
insertions and directions around the incumbent's unchanged service skeleton.
The final candidate is accepted only after the ordinary decoder confirms a
strict improvement.  This was worse at one second on the 10 x 10 group but
improved two independent controlled five-second runs on that group.  It
remains opt-in pending a full-suite confirmation.

Three additional, deliberately opt-in neighbourhoods were explored from
load-dependent VRP and arc-routing ideas.  None changes the default solver.

- `--enable-load-aware-ruin` reserves a final window, removes the currently
  most expensive services after multiplying their route contribution by their
  stage load, then applies regret reinsertion.  It did not improve the
  controlled one-second result across three seeds.
- `--enable-route-block-relocate` samples a costly contiguous block of decoded
  services and tries assigning it to another stage.  It was worse in the
  controlled one-second test.
- `--enable-sequence-split-vnd` reserves a final window, flattens the decoded
  service order, exactly chooses the best *contiguous* split of that sequence
  across the fixed delivery stages, and optionally perturbs the sequence with
  swap/relocate/reverse/block moves.  Contiguity makes this a restricted exact
  subproblem, not an exact LDdGRP solve.

The final windows are part of the reported configuration: use
`--load-aware-final-seconds`, `--load-aware-final-passes`, and
`--sequence-split-final-seconds` to reproduce an ablation. The giant-tour
window is controlled by `--giant-tour-final-seconds`. Their enabled sum must
be less than `--seconds`; the runner checks this and resume compatibility.
The corrected two-seed 10x10 five-second sequence/split experiment is
promising but mixed (mean paired objective change -0.607%, 10 wins / 5 ties /
9 losses); at one second it is worse pooled.  It therefore remains opt-in,
pending full-suite, multi-seed evidence.  See `LDDGRP_HANDOFF.md` for the
artifacts and complete protocol.

`--enable-giant-tour-vnd` is a newer opt-in long-budget intensifier. It lifts
the current route certificate into one permutation of delivery tasks and
required-edge services, scores a fixed permutation exactly with a two-state
orientation dynamic programme, then scans delivery/edge relocate and short
Or-opt moves. A candidate is retained only after it is projected back into
stage certificates, compared stage-by-stage with the ordinary decoder, and
validated. This removes the representation barrier that previously made a
delivery move leave nearby service edges behind on unrelated stage indices.

The setting below is the current controlled three-second ablation, not a new
default:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
py -3.13 -m lddgrp.benchmarks work/lddgrp-benchmarks work/LDdGRP-results.txt `
  --pattern 'LDdGRP10_10_*.dat' --seconds 3 --iterations 6000 --seed 7 `
  --large-stage-local-search-passes 12 --enable-giant-tour-vnd `
  --giant-tour-final-seconds 0.80 --giant-tour-max-passes 2 `
  --giant-tour-max-block-size 2 `
  --output outputs/experiment.json
```

On the 12 official 10x10 instances, matched three-second runs at seeds 7, 19,
and 31 gave 36 improvements and no ties or losses versus VND-12 (pooled mean
paired objective change -2.237%). One 7x7 seed is also encouraging; 4x4 did
not improve. This is strong targeted evidence, but **not** an all-instance
result, so leave the one-second default unchanged.

## Research next step

The `Candidate` representation and the `EdgeStageRepairOracle` protocol are
intentional extension points.  The included enumerator is exact only within a
small, fixed delivery-order/two-stage neighbourhood.  To turn this into the
recommended ALNS matheuristic, replace it with a time-limited exact LDdGRP
branch-and-cut repair model while fixing decisions outside a destroyed
neighbourhood.  This preserves the outer solver and makes the exact model an
intensification oracle.

