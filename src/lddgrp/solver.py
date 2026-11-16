"""A stage-aware ALNS for the discrete LDdGRP.

The search is deliberately separated from decoding.  This keeps every candidate
feasible at the high level and allows an exact MILP repair oracle to replace the
stage decoder later without redesigning the outer metaheuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import exp
from random import Random
from time import monotonic
from typing import Callable, Iterable, Literal

from .model import EPSILON, LDdGRPInstance
from .repair import EdgeStageRepairOracle, EnumerativeEdgeStageRepair


@dataclass(frozen=True, slots=True)
class Candidate:
    """A complete high-level LDdGRP decision.

    ``edge_stages[e]`` is zero-based and identifies the interval in which the
    required edge at index ``e`` is serviced.
    """

    delivery_order: tuple[int, ...]
    edge_stages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RouteStep:
    kind: Literal["deadhead", "service"]
    from_node: int
    to_node: int
    cost: float
    edge_index: int | None = None


@dataclass(frozen=True, slots=True)
class StageRoute:
    stage: int
    start_node: int
    end_node: int
    edge_indices: tuple[int, ...]
    steps: tuple[RouteStep, ...]
    base_cost: float
    multiplier: float
    weighted_cost: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    candidate: Candidate
    objective: float
    stages: tuple[StageRoute, ...]


@dataclass(frozen=True, slots=True)
class GiantTour:
    """A lossless interleaving of LDdGRP deliveries and required-edge services.

    Task identifiers ``0 .. m - 1`` identify required edges.  Identifiers
    ``m .. m + d - 1`` identify deliveries in ``instance.delivery_nodes``
    order, where ``m`` and ``d`` are the instance's required-edge and delivery
    counts.  The depot is implicit at both ends.  A delivery task terminates
    one service stage, so all preceding edge tasks belong to that stage.

    The representation retains a service order that ``Candidate`` deliberately
    does not store.  It is therefore used with a route certificate, never as a
    claim that the ordinary candidate decoder will reconstruct that order.
    """

    tasks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ALNSConfig:
    """Search controls.

    ``exact_stage_limit`` controls the per-stage bitmask DP, not global
    optimality.  Lower values trade solution quality for much faster search.
    """

    time_limit_seconds: float = 10.0
    max_iterations: int = 5_000
    exact_stage_limit: int = 10
    seed: int = 0
    initial_temperature_fraction: float = 0.02
    cooling_rate: float = 0.997
    adaptation_period: int = 50
    local_repair_max_edges: int = 7
    local_repair_seconds: float = 0.15
    initialization_fraction: float = 0.25
    large_stage_local_search_passes: int = 12
    search_large_stage_local_search_passes: int = 12
    finalization_fraction: float = 0.15
    stage_descent_cluster_size: int = 12
    stage_descent_seconds: float = 0.12
    stage_descent_passes: int = 1
    enable_stage_descent: bool = False
    spatial_repair_max_edges: int = 6
    spatial_repair_seconds: float = 0.12
    enable_spatial_repair: bool = False
    skeleton_repair_max_edges: int = 7
    skeleton_repair_seconds: float = 0.08
    enable_skeleton_repair: bool = False
    enable_delivery_order_dp: bool = False
    load_aware_ruin_max_edges: int = 6
    load_aware_ruin_seconds: float = 0.06
    load_aware_final_seconds: float = 0.08
    load_aware_final_passes: int = 1
    enable_load_aware_ruin: bool = False
    route_block_max_edges: int = 4
    route_block_seconds: float = 0.04
    enable_route_block_relocate: bool = False
    sequence_split_seconds: float = 0.10
    sequence_split_max_moves: int = 12
    sequence_split_final_seconds: float = 0.12
    enable_sequence_split_vnd: bool = False
    giant_tour_final_seconds: float = 0.80
    giant_tour_max_passes: int = 2
    giant_tour_max_block_size: int = 2
    enable_giant_tour_vnd: bool = False


class ALNSSolver:
    """Feasible-candidate ALNS with exact small-stage decoding and local repair."""

    def __init__(
        self,
        instance: LDdGRPInstance,
        config: ALNSConfig | None = None,
        repair_oracle: EdgeStageRepairOracle | None = None,
    ) -> None:
        self.instance = instance
        self.config = config or ALNSConfig()
        if self.config.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive.")
        if self.config.exact_stage_limit < 0:
            raise ValueError("exact_stage_limit must be non-negative.")
        if self.config.local_repair_max_edges < 1:
            raise ValueError("local_repair_max_edges must be at least one.")
        if self.config.local_repair_seconds <= 0:
            raise ValueError("local_repair_seconds must be positive.")
        if not 0 < self.config.initialization_fraction <= 1:
            raise ValueError("initialization_fraction must be in (0, 1].")
        if self.config.large_stage_local_search_passes < 0:
            raise ValueError("large_stage_local_search_passes must be non-negative.")
        if self.config.search_large_stage_local_search_passes < 0:
            raise ValueError("search_large_stage_local_search_passes must be non-negative.")
        if not 0 < self.config.finalization_fraction < 1:
            raise ValueError("finalization_fraction must be in (0, 1).")
        if self.config.stage_descent_cluster_size < 1:
            raise ValueError("stage_descent_cluster_size must be at least one.")
        if self.config.stage_descent_seconds <= 0:
            raise ValueError("stage_descent_seconds must be positive.")
        if self.config.stage_descent_passes < 1:
            raise ValueError("stage_descent_passes must be at least one.")
        if self.config.spatial_repair_max_edges < 1:
            raise ValueError("spatial_repair_max_edges must be at least one.")
        if self.config.spatial_repair_seconds <= 0:
            raise ValueError("spatial_repair_seconds must be positive.")
        if self.config.skeleton_repair_max_edges < 1:
            raise ValueError("skeleton_repair_max_edges must be at least one.")
        if self.config.skeleton_repair_seconds <= 0:
            raise ValueError("skeleton_repair_seconds must be positive.")
        if self.config.load_aware_ruin_max_edges < 1:
            raise ValueError("load_aware_ruin_max_edges must be at least one.")
        if self.config.load_aware_ruin_seconds <= 0:
            raise ValueError("load_aware_ruin_seconds must be positive.")
        if self.config.load_aware_final_seconds <= 0:
            raise ValueError("load_aware_final_seconds must be positive.")
        if self.config.load_aware_final_passes < 1:
            raise ValueError("load_aware_final_passes must be at least one.")
        if self.config.route_block_max_edges < 1:
            raise ValueError("route_block_max_edges must be at least one.")
        if self.config.route_block_seconds <= 0:
            raise ValueError("route_block_seconds must be positive.")
        if self.config.sequence_split_seconds <= 0:
            raise ValueError("sequence_split_seconds must be positive.")
        if self.config.sequence_split_max_moves < 0:
            raise ValueError("sequence_split_max_moves must be non-negative.")
        if self.config.sequence_split_final_seconds <= 0:
            raise ValueError("sequence_split_final_seconds must be positive.")
        if self.config.giant_tour_final_seconds <= 0:
            raise ValueError("giant_tour_final_seconds must be positive.")
        if self.config.giant_tour_max_passes < 1:
            raise ValueError("giant_tour_max_passes must be at least one.")
        if self.config.giant_tour_max_block_size < 1:
            raise ValueError("giant_tour_max_block_size must be at least one.")
        reserved_final_seconds = (
            (self.config.load_aware_final_seconds if self.config.enable_load_aware_ruin else 0.0)
            + (self.config.sequence_split_final_seconds if self.config.enable_sequence_split_vnd else 0.0)
            + (self.config.giant_tour_final_seconds if self.config.enable_giant_tour_vnd else 0.0)
        )
        if reserved_final_seconds >= self.config.time_limit_seconds:
            raise ValueError("A final-repair time reservation must be shorter than the total time limit.")
        self.rng = Random(self.config.seed)
        self._stage_cache: dict[tuple[int, int, int, tuple[int, ...]], tuple[RouteStep, ...]] = {}
        self.repair_oracle = repair_oracle or EnumerativeEdgeStageRepair(self.config.local_repair_max_edges)
        self._search_deadline: float | None = None
        self._active_large_stage_local_search_passes = self.config.large_stage_local_search_passes

    def solve(self) -> Evaluation:
        """Run ALNS and return the best validated solution found."""

        started = monotonic()
        deadline = started + self.config.time_limit_seconds
        construction_deadline = min(
            deadline, started + self.config.time_limit_seconds * self.config.initialization_fraction
        )
        self._search_deadline = construction_deadline
        self._active_large_stage_local_search_passes = self.config.large_stage_local_search_passes
        current = self.initial_candidate()
        self._active_large_stage_local_search_passes = self.config.search_large_stage_local_search_passes
        current_eval = self.evaluate(current)
        best = current_eval
        temperature = max(1.0, best.objective * self.config.initial_temperature_fraction)
        search_deadline = (
            deadline
            if self.config.search_large_stage_local_search_passes == self.config.large_stage_local_search_passes
            else started + self.config.time_limit_seconds * (1.0 - self.config.finalization_fraction)
        )
        final_repair_seconds = 0.0
        if self.config.enable_load_aware_ruin:
            # A load-aware regret rebuild performs several true decoded
            # insertion evaluations.  Sampling it like a cheap ALNS mutation
            # starves the outer search under a short wall-clock budget.  Keep
            # a bounded final intensification window instead.
            final_repair_seconds += self.config.load_aware_final_seconds
        if self.config.enable_sequence_split_vnd:
            final_repair_seconds += self.config.sequence_split_final_seconds
        if self.config.enable_giant_tour_vnd:
            final_repair_seconds += self._giant_tour_time_cap()
        if final_repair_seconds:
            search_deadline = min(
                search_deadline,
                deadline - final_repair_seconds,
            )
        self._search_deadline = search_deadline

        operators: dict[str, Callable[[Candidate], Candidate]] = {
            "delivery_relocate": self._delivery_relocate,
            "delivery_swap": self._delivery_swap,
            "delivery_reverse": self._delivery_reverse,
            "delivery_ruin_recreate": self._delivery_ruin_recreate,
            "edge_reassign": self._edge_reassign,
            "edge_best_reassign": self._edge_best_reassign,
            "edge_cluster_reassign": self._edge_cluster_reassign,
            "edge_cluster_recreate": self._edge_cluster_recreate,
            "stage_ruin_repair": self._stage_ruin_repair,
        }
        if self.config.enable_stage_descent:
            operators["edge_cluster_stage_descent"] = self._edge_cluster_stage_descent
        if self.config.enable_spatial_repair:
            operators["spatial_two_stage_exact_repair"] = self._spatial_two_stage_exact_repair
        if self.config.enable_skeleton_repair:
            operators["skeleton_two_stage_exact_repair"] = self._skeleton_two_stage_exact_repair
        if self.config.enable_delivery_order_dp:
            operators["delivery_order_dp"] = self._delivery_order_dp
        if self.config.enable_route_block_relocate:
            operators["route_block_relocate"] = self._route_block_relocate
        weights = {name: 1.0 for name in operators}
        rewards = {name: 0.0 for name in operators}
        uses = {name: 0 for name in operators}

        for iteration in range(1, self.config.max_iterations + 1):
            if monotonic() >= search_deadline:
                break
            name = self._weighted_choice(weights)
            proposal = operators[name](current)
            proposal_eval = self.evaluate(proposal)
            uses[name] += 1
            delta = proposal_eval.objective - current_eval.objective
            accepted = delta <= EPSILON or self.rng.random() < exp(-delta / max(temperature, EPSILON))

            if proposal_eval.objective + EPSILON < best.objective:
                best = proposal_eval
                rewards[name] += 6.0
            elif proposal_eval.objective + EPSILON < current_eval.objective:
                rewards[name] += 3.0
            elif accepted:
                rewards[name] += 0.5

            if accepted:
                current, current_eval = proposal, proposal_eval
            temperature *= self.config.cooling_rate

            if iteration % self.config.adaptation_period == 0:
                for operator_name in operators:
                    observed = rewards[operator_name] / max(1, uses[operator_name])
                    weights[operator_name] = max(0.1, 0.8 * weights[operator_name] + 0.2 * observed)
                    rewards[operator_name] = 0.0
                    uses[operator_name] = 0

        self._active_large_stage_local_search_passes = self.config.large_stage_local_search_passes
        if self.config.enable_giant_tour_vnd:
            self._search_deadline = min(deadline, monotonic() + self._giant_tour_time_cap())
            giant_tour_evaluation = self._giant_tour_vnd(best)
            if giant_tour_evaluation.objective + EPSILON < best.objective:
                best = giant_tour_evaluation
        if self.config.enable_load_aware_ruin:
            # ``_search_deadline`` still contains the ALNS cutoff, which has
            # normally passed.  Give each final repair its reserved window;
            # otherwise it would silently return before doing any work.
            self._search_deadline = min(deadline, monotonic() + self.config.load_aware_final_seconds)
            for _ in range(self.config.load_aware_final_passes):
                repaired = self._load_aware_ruin_recreate(best.candidate)
                repaired_evaluation = self.evaluate(repaired)
                if repaired_evaluation.objective + EPSILON < best.objective:
                    best = repaired_evaluation
                else:
                    break
        if self.config.enable_sequence_split_vnd:
            self._search_deadline = min(deadline, monotonic() + self.config.sequence_split_final_seconds)
            repaired = self._sequence_split_vnd(best.candidate)
            repaired_evaluation = self.evaluate(repaired)
            if repaired_evaluation.objective + EPSILON < best.objective:
                best = repaired_evaluation
        final = self.evaluate(best.candidate)
        self.validate_evaluation(final)
        self._search_deadline = None
        return final

    def initial_candidate(self) -> Candidate:
        """Construct a demand-aware delivery order and coupled edge allocation."""

        remaining = set(self.instance.delivery_nodes)
        order: list[int] = []
        current = self.instance.depot
        while remaining:
            # Larger demands are attractive earlier; randomisation supplies starts
            # with diversity without ever compromising feasibility.
            def score(node: int) -> float:
                demand = self.instance.delivery_demands[node]
                return (demand + 1.0) / (1.0 + self.instance.cost(current, node)) * self.rng.uniform(0.85, 1.15)

            chosen = max(remaining, key=score)
            order.append(chosen)
            remaining.remove(chosen)
            current = chosen

        order_tuple = tuple(order)
        multipliers = self._stage_multipliers(order_tuple)
        starts = (self.instance.depot, *order_tuple)
        ends = (*order_tuple, self.instance.depot)
        groups: list[list[int]] = [[] for _ in range(self.instance.stage_count)]

        # Assigning an edge from its own endpoint-to-endpoint estimate makes
        # every other edge invisible, and tends to pile service into the last
        # low-load stage.  Instead, recompute the *marginal decoded route
        # cost* in each stage.  This construction is still polynomial and its
        # only exact work is the normal small-stage decoder.
        for edge_index in sorted(
            range(len(self.instance.required_edges)),
            key=lambda index: self.instance.required_edges[index].service_cost,
            reverse=True,
        ):
            if self._deadline_reached():
                # Finish a valid solution cheaply if a very small global time
                # budget is consumed by the coupled construction.  The outer
                # search will either improve it (if time remains) or return a
                # certified feasible result.
                edge = self.instance.required_edges[edge_index]

                def fallback_incremental(stage: int) -> float:
                    forward = (
                        self.instance.cost(starts[stage], edge.u)
                        + edge.service_cost
                        + self.instance.cost(edge.v, ends[stage])
                    )
                    backward = (
                        self.instance.cost(starts[stage], edge.v)
                        + edge.service_cost
                        + self.instance.cost(edge.u, ends[stage])
                    )
                    direct = self.instance.cost(starts[stage], ends[stage])
                    return multipliers[stage] * (min(forward, backward) - direct)

                groups[min(range(self.instance.stage_count), key=fallback_incremental)].append(edge_index)
                continue

            def incremental(stage: int) -> float:
                before = self._stage_base_cost(starts[stage], ends[stage], tuple(groups[stage]))
                after = self._stage_base_cost(
                    starts[stage], ends[stage], tuple((*groups[stage], edge_index))
                )
                return multipliers[stage] * (after - before)

            chosen_stage = min(range(self.instance.stage_count), key=incremental)
            groups[chosen_stage].append(edge_index)
        stages = [0] * len(self.instance.required_edges)
        for stage, edge_indices in enumerate(groups):
            for edge_index in edge_indices:
                stages[edge_index] = stage
        candidate = Candidate(order_tuple, tuple(stages))
        if self.config.enable_delivery_order_dp:
            candidate = self._delivery_order_dp(candidate)
        self.validate_candidate(candidate)
        return candidate

    def evaluate(self, candidate: Candidate) -> Evaluation:
        """Decode and cost a candidate without trusting any stored objective."""

        self.validate_candidate(candidate)
        groups: list[list[int]] = [[] for _ in range(self.instance.stage_count)]
        for edge_index, stage in enumerate(candidate.edge_stages):
            groups[stage].append(edge_index)

        stage_starts = (self.instance.depot, *candidate.delivery_order)
        stage_ends = (*candidate.delivery_order, self.instance.depot)
        remaining_load = self.instance.total_demand
        decoded: list[StageRoute] = []
        objective = 0.0

        for stage, edge_indices in enumerate(groups):
            steps = self._decode_stage(stage_starts[stage], stage_ends[stage], tuple(edge_indices))
            base_cost = sum(step.cost for step in steps)
            multiplier = self.instance.curb_weight + remaining_load
            weighted_cost = multiplier * (base_cost + self.instance.landing_takeoff_cost)
            decoded.append(
                StageRoute(
                    stage=stage,
                    start_node=stage_starts[stage],
                    end_node=stage_ends[stage],
                    edge_indices=tuple(edge_indices),
                    steps=steps,
                    base_cost=base_cost,
                    multiplier=multiplier,
                    weighted_cost=weighted_cost,
                )
            )
            objective += weighted_cost
            if stage < len(candidate.delivery_order):
                remaining_load -= self.instance.delivery_demands[stage_ends[stage]]

        return Evaluation(candidate=candidate, objective=objective, stages=tuple(decoded))

    def validate_candidate(self, candidate: Candidate) -> None:
        if len(candidate.delivery_order) != len(self.instance.delivery_nodes):
            raise ValueError("Candidate does not contain every delivery node exactly once.")
        if set(candidate.delivery_order) != set(self.instance.delivery_nodes):
            raise ValueError("Candidate delivery_order is not a permutation of delivery nodes.")
        if len(candidate.edge_stages) != len(self.instance.required_edges):
            raise ValueError("Candidate does not assign every required edge.")
        if any(stage < 0 or stage >= self.instance.stage_count for stage in candidate.edge_stages):
            raise ValueError("Candidate contains an invalid edge stage.")

    def validate_evaluation(self, evaluation: Evaluation) -> None:
        """Verify the route certificate and recompute the exact displayed cost."""

        self.validate_candidate(evaluation.candidate)
        serviced: list[int] = []
        recomputed = 0.0
        expected_load = self.instance.total_demand
        for stage in evaluation.stages:
            if not stage.steps:
                raise ValueError("Every stage must contain at least its direct endpoint connection.")
            current = stage.start_node
            base = 0.0
            for step in stage.steps:
                if step.from_node != current:
                    raise ValueError("Stage certificate is not a contiguous walk.")
                if step.kind == "deadhead":
                    expected = self.instance.cost(step.from_node, step.to_node)
                    if step.edge_index is not None:
                        raise ValueError("A deadheading step cannot identify a serviced edge.")
                else:
                    if step.edge_index is None:
                        raise ValueError("A service step must identify its required edge.")
                    if evaluation.candidate.edge_stages[step.edge_index] != stage.stage:
                        raise ValueError("A service step is assigned to a different candidate stage.")
                    edge = self.instance.required_edges[step.edge_index]
                    if {step.from_node, step.to_node} != {edge.u, edge.v}:
                        raise ValueError("A service step does not traverse its required edge.")
                    expected = edge.service_cost
                    serviced.append(step.edge_index)
                if abs(step.cost - expected) > 1e-7:
                    raise ValueError("A route step has an invalid cost.")
                base += step.cost
                current = step.to_node
            if current != stage.end_node or abs(base - stage.base_cost) > 1e-7:
                raise ValueError("A stage certificate has an invalid endpoint or base cost.")
            if abs(stage.multiplier - (self.instance.curb_weight + expected_load)) > 1e-7:
                raise ValueError("A stage uses an invalid load multiplier.")
            recomputed += stage.multiplier * (base + self.instance.landing_takeoff_cost)
            if stage.stage < len(evaluation.candidate.delivery_order):
                expected_load -= self.instance.delivery_demands[stage.end_node]
        if sorted(serviced) != list(range(len(self.instance.required_edges))):
            raise ValueError("The certificate does not service every required edge exactly once.")
        if abs(recomputed - evaluation.objective) > 1e-7:
            raise ValueError("The reported objective does not equal the certificate cost.")

    def _validate_giant_tour(self, tour: GiantTour) -> None:
        """Check the task encoding and one-use-per-task invariant."""

        task_count = len(self.instance.required_edges) + len(self.instance.delivery_nodes)
        if len(tour.tasks) != task_count:
            raise ValueError("A giant tour must contain every delivery and required edge exactly once.")
        if any(not isinstance(task, int) for task in tour.tasks) or set(tour.tasks) != set(range(task_count)):
            raise ValueError("A giant tour contains an invalid, missing, or duplicate task identifier.")

    def _giant_task_endpoints(self, task: int, orientation: int) -> tuple[int, int]:
        """Return a task's entry and exit node under one of its two states."""

        edge_count = len(self.instance.required_edges)
        if task < edge_count:
            edge = self.instance.required_edges[task]
            return (edge.u, edge.v) if orientation == 0 else (edge.v, edge.u)
        node = self.instance.delivery_nodes[task - edge_count]
        return node, node

    def _giant_task_delivery_node(self, task: int) -> int | None:
        """Return the delivery node represented by ``task``, if any."""

        edge_count = len(self.instance.required_edges)
        if task < edge_count:
            return None
        return self.instance.delivery_nodes[task - edge_count]

    def giant_tour_from_evaluation(self, evaluation: Evaluation) -> GiantTour:
        """Lift a valid route certificate into its lossless giant-tour order.

        The order of service steps in every decoded stage is retained.  This is
        the only safe lift from ``Candidate``: a candidate stores an edge-stage
        label but intentionally discards within-stage service order.
        """

        self.validate_evaluation(evaluation)
        edge_count = len(self.instance.required_edges)
        delivery_task = {
            node: edge_count + position for position, node in enumerate(self.instance.delivery_nodes)
        }
        tasks: list[int] = []
        for stage in evaluation.stages:
            for step in stage.steps:
                if step.kind == "service":
                    if step.edge_index is None:  # Defensive; validation above already rejects this.
                        raise ValueError("A service step lacks its required-edge index.")
                    tasks.append(step.edge_index)
            if stage.stage < len(evaluation.candidate.delivery_order):
                tasks.append(delivery_task[stage.end_node])
        tour = GiantTour(tuple(tasks))
        self._validate_giant_tour(tour)
        return tour

    def _giant_task_weight_after(self, task: int, weight_before: float) -> float:
        """Return the remaining vehicle weight after one giant-tour task."""

        delivery = self._giant_task_delivery_node(task)
        return weight_before if delivery is None else weight_before - self.instance.delivery_demands[delivery]

    def _giant_tour_time_cap(self) -> float:
        """Scale the giant-tour final window with its quadratic neighbourhood.

        ``giant_tour_final_seconds`` is a cap calibrated for roughly 100 tasks,
        the largest official instances. Smaller instances need markedly less
        time for the same bounded relocate/Or-opt scan, so reserving the full
        cap would needlessly starve ALNS. This is a budget allocation rule, not
        a complexity or speed guarantee.
        """

        task_count = len(self.instance.required_edges) + len(self.instance.delivery_nodes)
        relative_quadratic_work = (task_count / 100.0) ** 2
        return min(self.config.giant_tour_final_seconds, max(0.01, self.config.giant_tour_final_seconds * relative_quadratic_work))

    def _advance_giant_dp(
        self,
        previous: tuple[float, float],
        previous_task: int | None,
        task: int,
        weight_before: float,
    ) -> tuple[tuple[float, float], float, tuple[int, int]]:
        """Advance a two-state giant-tour prefix DP through one task."""

        weight_after = self._giant_task_weight_after(task, weight_before)
        delivery = self._giant_task_delivery_node(task)
        intrinsic = (
            weight_after * self.instance.landing_takeoff_cost
            if delivery is not None
            else weight_before * self.instance.required_edges[task].service_cost
        )
        current: list[float] = [float("inf"), float("inf")]
        parents: list[int] = [0, 0]
        for orientation in (0, 1):
            entry, _ = self._giant_task_endpoints(task, orientation)
            best_parent = min(
                (0, 1),
                key=lambda prior_orientation: previous[prior_orientation]
                + weight_before
                * self.instance.cost(
                    self.instance.depot
                    if previous_task is None
                    else self._giant_task_endpoints(previous_task, prior_orientation)[1],
                    entry,
                ),
            )
            previous_end = (
                self.instance.depot
                if previous_task is None
                else self._giant_task_endpoints(previous_task, best_parent)[1]
            )
            current[orientation] = (
                previous[best_parent] + weight_before * self.instance.cost(previous_end, entry) + intrinsic
            )
            parents[orientation] = best_parent
        return (current[0], current[1]), weight_after, (parents[0], parents[1])

    def _retreat_giant_dp(
        self,
        task: int,
        weight_before: float,
        next_task: int | None,
        suffix: tuple[float, float] | None,
    ) -> tuple[float, float]:
        """Return the two-state suffix DP beginning at ``task``.

        ``suffix`` includes the intrinsic cost of ``next_task``.  Passing no
        next task represents the final weighted return to the depot.
        """

        weight_after = self._giant_task_weight_after(task, weight_before)
        delivery = self._giant_task_delivery_node(task)
        intrinsic = (
            weight_after * self.instance.landing_takeoff_cost
            if delivery is not None
            else weight_before * self.instance.required_edges[task].service_cost
        )
        values: list[float] = [float("inf"), float("inf")]
        for orientation in (0, 1):
            _, exit_node = self._giant_task_endpoints(task, orientation)
            if next_task is None:
                values[orientation] = intrinsic + weight_after * self.instance.cost(exit_node, self.instance.depot)
            else:
                assert suffix is not None
                values[orientation] = intrinsic + min(
                    weight_after
                    * self.instance.cost(exit_node, self._giant_task_endpoints(next_task, next_orientation)[0])
                    + suffix[next_orientation]
                    for next_orientation in (0, 1)
                )
        return values[0], values[1]

    def _giant_tour_orientation_dp(
        self, tour: GiantTour, *, validate: bool = True
    ) -> tuple[float, tuple[int, ...]]:
        """Return the exact fixed-order score and one optimal edge orientation.

        This is intentionally separate from certificate construction so local
        search can score trusted task permutations without allocating route
        objects. The optional validation may only be skipped by internal moves
        that preserve a previously validated permutation.
        """

        if validate:
            self._validate_giant_tour(tour)
        tasks = tour.tasks
        weight = self.instance.curb_weight + self.instance.total_demand
        previous = (weight * self.instance.landing_takeoff_cost, float("inf"))
        parents: list[tuple[int, int]] = []
        for position, task in enumerate(tasks):
            previous, weight, parent = self._advance_giant_dp(
                previous,
                None if position == 0 else tasks[position - 1],
                task,
                weight,
            )
            parents.append(parent)

        if not tasks:
            return weight * self.instance.landing_takeoff_cost, ()
        final_orientation = min(
            (0, 1),
            key=lambda orientation: previous[orientation]
            + weight * self.instance.cost(self._giant_task_endpoints(tasks[-1], orientation)[1], self.instance.depot),
        )
        fixed_objective = previous[final_orientation] + weight * self.instance.cost(
            self._giant_task_endpoints(tasks[-1], final_orientation)[1], self.instance.depot
        )
        orientations = [0] * len(tasks)
        orientation = final_orientation
        for position in range(len(tasks) - 1, -1, -1):
            orientations[position] = orientation
            orientation = parents[position][orientation]
        return fixed_objective, tuple(orientations)

    def _giant_tour_dp_tables(
        self, tour: GiantTour
    ) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
        """Build exact prefix/suffix orientation tables for one task order."""

        self._validate_giant_tour(tour)
        tasks = tour.tasks
        weight = self.instance.curb_weight + self.instance.total_demand
        initial = (weight * self.instance.landing_takeoff_cost, float("inf"))
        weights_before: list[float] = []
        prefixes: list[tuple[float, float]] = []
        previous = initial
        for position, task in enumerate(tasks):
            weights_before.append(weight)
            previous, weight, _ = self._advance_giant_dp(
                previous,
                None if position == 0 else tasks[position - 1],
                task,
                weight,
            )
            prefixes.append(previous)

        suffixes: list[tuple[float, float]] = [(float("inf"), float("inf")) for _ in tasks]
        next_task: int | None = None
        suffix: tuple[float, float] | None = None
        for position in range(len(tasks) - 1, -1, -1):
            suffix = self._retreat_giant_dp(tasks[position], weights_before[position], next_task, suffix)
            suffixes[position] = suffix
            next_task = tasks[position]
        return tuple(weights_before), tuple(prefixes), tuple(suffixes)

    def _stitch_giant_dp(
        self,
        prefix: tuple[float, float],
        previous_task: int,
        weight_before_next: float,
        next_task: int | None,
        suffix: tuple[float, float] | None,
    ) -> float:
        """Connect a re-scored giant-tour prefix to an unchanged suffix."""

        if next_task is None:
            return min(
                prefix[orientation]
                + weight_before_next
                * self.instance.cost(self._giant_task_endpoints(previous_task, orientation)[1], self.instance.depot)
                for orientation in (0, 1)
            )
        assert suffix is not None
        return min(
            prefix[orientation]
            + weight_before_next
            * self.instance.cost(
                self._giant_task_endpoints(previous_task, orientation)[1],
                self._giant_task_endpoints(next_task, next_orientation)[0],
            )
            + suffix[next_orientation]
            for orientation in (0, 1)
            for next_orientation in (0, 1)
        )

    def _best_giant_tour_relocate(
        self, tour: GiantTour, *, max_block_size: int = 1, deadline: float | None = None
    ) -> tuple[GiantTour, float]:
        """Find the exact fixed-order best relocate/short-Or-opt neighbour.

        Forward and backward relocations are scanned with prefix/suffix
        orientation tables. A block has at most ``max_block_size`` tasks. The
        returned score is the exact giant-tour score, before normal stage
        decoder dominance is applied.
        """

        if max_block_size < 1:
            raise ValueError("max_block_size must be at least one.")
        self._validate_giant_tour(tour)
        tasks = tour.tasks
        task_count = len(tasks)
        current_score, _ = self._giant_tour_orientation_dp(tour, validate=False)
        if task_count < 2:
            return tour, current_score
        weights_before, prefixes, suffixes = self._giant_tour_dp_tables(tour)
        initial = (
            (self.instance.curb_weight + self.instance.total_demand) * self.instance.landing_takeoff_cost,
            float("inf"),
        )
        best_score = current_score
        best_tasks = tasks

        for first in range(task_count):
            if deadline is not None and monotonic() >= deadline:
                break
            delivery_weight = 0.0
            for size in range(1, min(max_block_size, task_count - first) + 1):
                last = first + size - 1
                delivery = self._giant_task_delivery_node(tasks[last])
                if delivery is not None:
                    delivery_weight += self.instance.delivery_demands[delivery]
                block = tasks[first : last + 1]

                # Move the block later. The crossed tasks retain an additional
                # ``delivery_weight`` until the block is finally executed.
                if last + 1 < task_count:
                    middle_dp = initial if first == 0 else prefixes[first - 1]
                    middle_previous = None if first == 0 else tasks[first - 1]
                    for insertion_after in range(last + 1, task_count):
                        crossed = tasks[insertion_after]
                        middle_dp, middle_weight, _ = self._advance_giant_dp(
                            middle_dp,
                            middle_previous,
                            crossed,
                            weights_before[insertion_after] + delivery_weight,
                        )
                        middle_previous = crossed
                        block_dp = middle_dp
                        block_previous = crossed
                        block_weight = middle_weight
                        for block_task in block:
                            block_dp, block_weight, _ = self._advance_giant_dp(
                                block_dp, block_previous, block_task, block_weight
                            )
                            block_previous = block_task
                        score = self._stitch_giant_dp(
                            block_dp,
                            block_previous,
                            block_weight,
                            tasks[insertion_after + 1] if insertion_after + 1 < task_count else None,
                            suffixes[insertion_after + 1] if insertion_after + 1 < task_count else None,
                        )
                        if score + EPSILON < best_score:
                            best_score = score
                            best_tasks = (
                                tasks[:first]
                                + tasks[last + 1 : insertion_after + 1]
                                + block
                                + tasks[insertion_after + 1 :]
                            )

                # Move the block earlier. The crossed tasks see the block's
                # delivery demand removed before they are traversed.
                if first > 0:
                    shifted_suffixes: list[tuple[float, float]] = [
                        (float("inf"), float("inf")) for _ in range(first)
                    ]
                    next_task = tasks[last + 1] if last + 1 < task_count else None
                    shifted_suffix = suffixes[last + 1] if last + 1 < task_count else None
                    for position in range(first - 1, -1, -1):
                        shifted_suffix = self._retreat_giant_dp(
                            tasks[position],
                            weights_before[position] - delivery_weight,
                            next_task,
                            shifted_suffix,
                        )
                        shifted_suffixes[position] = shifted_suffix
                        next_task = tasks[position]
                    for insertion_before in range(first):
                        block_dp = initial if insertion_before == 0 else prefixes[insertion_before - 1]
                        block_previous = None if insertion_before == 0 else tasks[insertion_before - 1]
                        block_weight = weights_before[insertion_before]
                        for block_task in block:
                            block_dp, block_weight, _ = self._advance_giant_dp(
                                block_dp, block_previous, block_task, block_weight
                            )
                            block_previous = block_task
                        score = self._stitch_giant_dp(
                            block_dp,
                            block_previous,
                            block_weight,
                            tasks[insertion_before],
                            shifted_suffixes[insertion_before],
                        )
                        if score + EPSILON < best_score:
                            best_score = score
                            best_tasks = (
                                tasks[:insertion_before]
                                + block
                                + tasks[insertion_before:first]
                                + tasks[last + 1 :]
                            )

        best_tour = GiantTour(best_tasks)
        verified_score, _ = self._giant_tour_orientation_dp(best_tour, validate=False)
        if abs(verified_score - best_score) > 1e-7:
            raise ValueError("The incremental giant-tour relocate score disagrees with full evaluation.")
        return best_tour, verified_score

    def _giant_tour_vnd(self, incumbent: Evaluation) -> Evaluation:
        """Run bounded exact-scored giant-tour relocate/Or-opt descent.

        The inner scan uses the fixed-tour dynamic-programme score. Its best
        proposed move is then certified with stage-by-stage decoder dominance
        before it can replace the incumbent. This deliberately performs at
        most one normal stage decode per VND pass, rather than per neighbour.
        """

        deadline = self._search_deadline if self._search_deadline is not None else float("inf")
        current = incumbent
        for _ in range(self.config.giant_tour_max_passes):
            if monotonic() >= deadline:
                break
            tour = self.giant_tour_from_evaluation(current)
            current_fixed_score, _ = self._giant_tour_orientation_dp(tour, validate=False)
            proposal_tour, proposal_fixed_score = self._best_giant_tour_relocate(
                tour,
                max_block_size=self.config.giant_tour_max_block_size,
                deadline=deadline,
            )
            if proposal_fixed_score + EPSILON >= current_fixed_score or monotonic() >= deadline:
                break
            proposal = self.evaluate_giant_tour(proposal_tour, dominate_with_decoder=True)
            if proposal.objective + EPSILON >= current.objective:
                break
            current = proposal
        return current

    def evaluate_giant_tour(self, tour: GiantTour, *, dominate_with_decoder: bool = True) -> Evaluation:
        """Exactly evaluate one fixed task sequence and project its certificate.

        The two-state dynamic programme chooses every required-edge direction
        exactly for the supplied giant-tour task order. Deliveries terminate
        stages, so their position jointly determines the delivery permutation
        and all edge-stage labels. When ``dominate_with_decoder`` is true,
        each projected stage is compared independently with the normal decoder
        and the cheaper feasible certificate is retained. This can only lower
        the displayed objective; it does *not* make the fixed-tour DP globally
        exact for LDdGRP.
        """

        self._validate_giant_tour(tour)
        tasks = tour.tasks
        edge_count = len(self.instance.required_edges)

        delivery_order: list[int] = []
        edge_stages: list[int | None] = [None] * edge_count
        stage = 0
        for task in tasks:
            delivery = self._giant_task_delivery_node(task)
            if delivery is None:
                edge_stages[task] = stage
            else:
                delivery_order.append(delivery)
                stage += 1
        if any(value is None for value in edge_stages):  # Defensive; tour validation proves coverage.
            raise ValueError("A giant tour did not assign every required edge to a stage.")
        candidate = Candidate(tuple(delivery_order), tuple(int(value) for value in edge_stages))
        self.validate_candidate(candidate)

        fixed_objective, orientations = self._giant_tour_orientation_dp(tour, validate=False)

        multipliers = self._stage_multipliers(candidate.delivery_order)
        fixed_stages: list[StageRoute] = []
        current_node = self.instance.depot
        stage_start = self.instance.depot
        stage_edges: list[int] = []
        stage_steps: list[RouteStep] = []
        stage = 0

        def finish_stage(end_node: int) -> None:
            nonlocal current_node
            if current_node != end_node:
                stage_steps.append(
                    RouteStep("deadhead", current_node, end_node, self.instance.cost(current_node, end_node))
                )
            if not stage_steps:
                stage_steps.append(RouteStep("deadhead", stage_start, end_node, self.instance.cost(stage_start, end_node)))
            base_cost = sum(step.cost for step in stage_steps)
            fixed_stages.append(
                StageRoute(
                    stage=stage,
                    start_node=stage_start,
                    end_node=end_node,
                    edge_indices=tuple(stage_edges),
                    steps=tuple(stage_steps),
                    base_cost=base_cost,
                    multiplier=multipliers[stage],
                    weighted_cost=multipliers[stage] * (base_cost + self.instance.landing_takeoff_cost),
                )
            )
            current_node = end_node

        for task, orientation in zip(tasks, orientations, strict=True):
            delivery = self._giant_task_delivery_node(task)
            if delivery is not None:
                finish_stage(delivery)
                stage += 1
                stage_start = delivery
                current_node = delivery
                stage_edges = []
                stage_steps = []
                continue
            service_start, service_end = self._giant_task_endpoints(task, orientation)
            if current_node != service_start:
                stage_steps.append(
                    RouteStep("deadhead", current_node, service_start, self.instance.cost(current_node, service_start))
                )
            edge = self.instance.required_edges[task]
            stage_steps.append(RouteStep("service", service_start, service_end, edge.service_cost, task))
            stage_edges.append(task)
            current_node = service_end
        finish_stage(self.instance.depot)

        fixed_evaluation = Evaluation(
            candidate=candidate,
            objective=sum(route.weighted_cost for route in fixed_stages),
            stages=tuple(fixed_stages),
        )
        if abs(fixed_evaluation.objective - fixed_objective) > 1e-7:
            raise ValueError("The giant-tour dynamic-programme cost disagrees with its route certificate.")
        self.validate_evaluation(fixed_evaluation)
        if not dominate_with_decoder:
            return fixed_evaluation

        dominated_stages: list[StageRoute] = []
        for route in fixed_evaluation.stages:
            decoder_steps = self._decode_stage(route.start_node, route.end_node, route.edge_indices)
            decoder_base = sum(step.cost for step in decoder_steps)
            if decoder_base + EPSILON < route.base_cost:
                retained_steps = decoder_steps
                retained_base = decoder_base
            else:
                retained_steps = route.steps
                retained_base = route.base_cost
                if retained_base + EPSILON < decoder_base:
                    key = (
                        route.start_node,
                        route.end_node,
                        self._active_large_stage_local_search_passes,
                        tuple(sorted(route.edge_indices)),
                    )
                    self._stage_cache[key] = retained_steps
            dominated_stages.append(
                StageRoute(
                    stage=route.stage,
                    start_node=route.start_node,
                    end_node=route.end_node,
                    edge_indices=route.edge_indices,
                    steps=retained_steps,
                    base_cost=retained_base,
                    multiplier=route.multiplier,
                    weighted_cost=route.multiplier * (retained_base + self.instance.landing_takeoff_cost),
                )
            )
        evaluation = Evaluation(
            candidate=candidate,
            objective=sum(route.weighted_cost for route in dominated_stages),
            stages=tuple(dominated_stages),
        )
        self.validate_evaluation(evaluation)
        return evaluation

    def _decode_stage(self, start: int, end: int, edge_indices: tuple[int, ...]) -> tuple[RouteStep, ...]:
        key = (start, end, self._active_large_stage_local_search_passes, tuple(sorted(edge_indices)))
        cached = self._stage_cache.get(key)
        if cached is not None:
            return cached
        if not edge_indices:
            result = (RouteStep("deadhead", start, end, self.instance.cost(start, end)),)
        elif len(edge_indices) <= self.config.exact_stage_limit:
            result = self._exact_stage_route(start, end, edge_indices)
        else:
            result = self._greedy_stage_route(start, end, edge_indices)
        self._stage_cache[key] = result
        return result

    def _stage_base_cost(self, start: int, end: int, edge_indices: tuple[int, ...]) -> float:
        """Return an exactly decoded or heuristic stage cost, as configured."""

        return sum(step.cost for step in self._decode_stage(start, end, edge_indices))

    def _exact_stage_route(self, start: int, end: int, edge_indices: tuple[int, ...]) -> tuple[RouteStep, ...]:
        """Held-Karp DP over edge order and the two possible orientations."""

        count = len(edge_indices)
        costs: dict[tuple[int, int, int], float] = {}
        parent: dict[tuple[int, int, int], tuple[int, int] | None] = {}

        def endpoints(local_edge: int, orientation: int) -> tuple[int, int]:
            edge = self.instance.required_edges[edge_indices[local_edge]]
            return (edge.u, edge.v) if orientation == 0 else (edge.v, edge.u)

        for edge_position in range(count):
            edge = self.instance.required_edges[edge_indices[edge_position]]
            for orientation in (0, 1):
                service_start, _ = endpoints(edge_position, orientation)
                key = (1 << edge_position, edge_position, orientation)
                costs[key] = self.instance.cost(start, service_start) + edge.service_cost
                parent[key] = None

        for mask in range(1, 1 << count):
            for last in range(count):
                if not mask & (1 << last):
                    continue
                for orientation in (0, 1):
                    key = (mask, last, orientation)
                    current_cost = costs.get(key)
                    if current_cost is None:
                        continue
                    _, last_end = endpoints(last, orientation)
                    for nxt in range(count):
                        if mask & (1 << nxt):
                            continue
                        edge = self.instance.required_edges[edge_indices[nxt]]
                        for next_orientation in (0, 1):
                            next_start, _ = endpoints(nxt, next_orientation)
                            next_key = (mask | (1 << nxt), nxt, next_orientation)
                            trial = current_cost + self.instance.cost(last_end, next_start) + edge.service_cost
                            if trial + EPSILON < costs.get(next_key, float("inf")):
                                costs[next_key] = trial
                                parent[next_key] = (last, orientation)

        full = (1 << count) - 1
        final_key = min(
            ((full, last, orientation) for last in range(count) for orientation in (0, 1)),
            key=lambda key: costs[key] + self.instance.cost(endpoints(key[1], key[2])[1], end),
        )
        order_orientations: list[tuple[int, int]] = []
        key: tuple[int, int, int] | None = final_key
        while key is not None:
            mask, last, orientation = key
            order_orientations.append((last, orientation))
            predecessor = parent[key]
            key = None if predecessor is None else (mask ^ (1 << last), predecessor[0], predecessor[1])
        order_orientations.reverse()
        return self._steps_from_order(start, end, edge_indices, order_orientations)

    def _greedy_stage_route(self, start: int, end: int, edge_indices: tuple[int, ...]) -> tuple[RouteStep, ...]:
        """Construct and improve a large stage with orientation-aware descent.

        The initial nearest-service order is fast but can contain expensive
        crossings.  The subsequent descent tests every 2-opt and one-edge
        relocation move in O(m^2) for a stage containing m required edges.  A
        move is evaluated on a feasible oriented route and then its directions
        are optimised exactly by the two-state dynamic programme.
        """

        return self._improve_stage_order(start, end, edge_indices, self._nearest_service_order(start, end, edge_indices))

    def _nearest_service_order(self, start: int, end: int, edge_indices: tuple[int, ...]) -> tuple[int, ...]:
        """Build the legacy nearest-service ordering for a stage."""

        unserved = set(range(len(edge_indices)))
        current = start
        order: list[int] = []
        while unserved:
            def choice(local_edge: int) -> tuple[float, int, int]:
                edge = self.instance.required_edges[edge_indices[local_edge]]
                forward = self.instance.cost(current, edge.u) + edge.service_cost + 0.15 * self.instance.cost(edge.v, end)
                backward = self.instance.cost(current, edge.v) + edge.service_cost + 0.15 * self.instance.cost(edge.u, end)
                if forward <= backward:
                    return forward, edge.u, edge.v
                return backward, edge.v, edge.u

            chosen = min(unserved, key=lambda local_edge: choice(local_edge)[0])
            _, _, current = choice(chosen)
            order.append(chosen)
            unserved.remove(chosen)
        return tuple(order)

    def _improve_stage_order(
        self, start: int, end: int, edge_indices: tuple[int, ...], initial_order: Iterable[int]
    ) -> tuple[RouteStep, ...]:
        """Apply bounded orientation-aware 2-opt and relocation descent."""

        sequence, orientations, current_cost = self._orient_stage_order(start, end, edge_indices, tuple(initial_order))
        for _ in range(self._active_large_stage_local_search_passes):
            two_opt = self._best_two_opt_move(start, end, edge_indices, sequence, orientations)
            relocate = self._best_relocate_move(start, end, edge_indices, sequence, orientations)
            move = min((move for move in (two_opt, relocate) if move is not None), key=lambda item: item[2], default=None)
            if move is None:
                break
            trial_sequence, _, _ = move
            trial_sequence, trial_orientations, actual_cost = self._orient_stage_order(
                start, end, edge_indices, trial_sequence
            )
            if actual_cost + EPSILON >= current_cost:
                # A move is only selected when its fixed-orientation version is
                # improving; this guard protects the decoder against numerical
                # noise before accepting it into the route certificate.
                break
            sequence, orientations = trial_sequence, trial_orientations
            current_cost = actual_cost
        return self._steps_from_order(start, end, edge_indices, zip(sequence, orientations, strict=True))

    def _orient_stage_order(
        self, start: int, end: int, edge_indices: tuple[int, ...], sequence: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...], float]:
        """Return the best orientation and its route cost for a fixed edge order."""

        steps = self._best_orientations_for_order(start, end, edge_indices, sequence)
        orientations: list[int] = []
        seen_sequence: list[int] = []
        local_index = {edge_index: index for index, edge_index in enumerate(edge_indices)}
        for step in steps:
            if step.kind != "service":
                continue
            assert step.edge_index is not None
            edge = self.instance.required_edges[step.edge_index]
            if step.from_node == edge.u and step.to_node == edge.v:
                orientation = 0
            elif step.from_node == edge.v and step.to_node == edge.u:
                orientation = 1
            else:  # Defensive: _best_orientations_for_order must service the edge.
                raise ValueError("A stage service step has an invalid orientation.")
            seen_sequence.append(local_index[step.edge_index])
            orientations.append(orientation)
        if tuple(seen_sequence) != sequence:
            raise ValueError("The stage decoder returned a different edge order.")
        return sequence, tuple(orientations), sum(step.cost for step in steps)

    def _oriented_endpoints(
        self, edge_indices: tuple[int, ...], local_edge: int, orientation: int
    ) -> tuple[int, int]:
        edge = self.instance.required_edges[edge_indices[local_edge]]
        return (edge.u, edge.v) if orientation == 0 else (edge.v, edge.u)

    def _best_two_opt_move(
        self,
        start: int,
        end: int,
        edge_indices: tuple[int, ...],
        sequence: tuple[int, ...],
        orientations: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], float] | None:
        """Find the best segment reversal, flipping each edge orientation.

        In a symmetric metric, reversing a segment and every serviced-edge
        direction preserves its internal deadheading cost.  Only the two
        boundary connections change, so all O(m^2) moves can be tested in O(1).
        """

        best: tuple[tuple[int, ...], tuple[int, ...], float] | None = None
        best_delta = 0.0
        count = len(sequence)
        for left in range(count - 1):
            previous = start if left == 0 else self._oriented_endpoints(
                edge_indices, sequence[left - 1], orientations[left - 1]
            )[1]
            first_start = self._oriented_endpoints(edge_indices, sequence[left], orientations[left])[0]
            for right in range(left + 1, count):
                last_end = self._oriented_endpoints(edge_indices, sequence[right], orientations[right])[1]
                following = end if right == count - 1 else self._oriented_endpoints(
                    edge_indices, sequence[right + 1], orientations[right + 1]
                )[0]
                delta = (
                    self.instance.cost(previous, last_end)
                    + self.instance.cost(first_start, following)
                    - self.instance.cost(previous, first_start)
                    - self.instance.cost(last_end, following)
                )
                if delta + EPSILON < best_delta:
                    new_sequence = sequence[:left] + tuple(reversed(sequence[left : right + 1])) + sequence[right + 1 :]
                    new_orientations = (
                        orientations[:left]
                        + tuple(1 - orientation for orientation in reversed(orientations[left : right + 1]))
                        + orientations[right + 1 :]
                    )
                    best_delta = delta
                    best = (new_sequence, new_orientations, delta)
        return best

    def _best_relocate_move(
        self,
        start: int,
        end: int,
        edge_indices: tuple[int, ...],
        sequence: tuple[int, ...],
        orientations: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], float] | None:
        """Find the best move that relocates one serviced edge in O(m^2)."""

        best: tuple[tuple[int, ...], tuple[int, ...], float] | None = None
        best_delta = 0.0
        for source, local_edge in enumerate(sequence):
            orientation = orientations[source]
            edge_start, edge_end = self._oriented_endpoints(edge_indices, local_edge, orientation)
            previous = start if source == 0 else self._oriented_endpoints(
                edge_indices, sequence[source - 1], orientations[source - 1]
            )[1]
            following = end if source == len(sequence) - 1 else self._oriented_endpoints(
                edge_indices, sequence[source + 1], orientations[source + 1]
            )[0]
            removal_delta = (
                self.instance.cost(previous, following)
                - self.instance.cost(previous, edge_start)
                - self.instance.cost(edge_end, following)
            )
            remaining_sequence = sequence[:source] + sequence[source + 1 :]
            remaining_orientations = orientations[:source] + orientations[source + 1 :]
            for destination in range(len(remaining_sequence) + 1):
                before = start if destination == 0 else self._oriented_endpoints(
                    edge_indices, remaining_sequence[destination - 1], remaining_orientations[destination - 1]
                )[1]
                after = end if destination == len(remaining_sequence) else self._oriented_endpoints(
                    edge_indices, remaining_sequence[destination], remaining_orientations[destination]
                )[0]
                delta = (
                    removal_delta
                    + self.instance.cost(before, edge_start)
                    + self.instance.cost(edge_end, after)
                    - self.instance.cost(before, after)
                )
                if delta + EPSILON < best_delta:
                    new_sequence = remaining_sequence[:destination] + (local_edge,) + remaining_sequence[destination:]
                    new_orientations = (
                        remaining_orientations[:destination] + (orientation,) + remaining_orientations[destination:]
                    )
                    best_delta = delta
                    best = (new_sequence, new_orientations, delta)
        return best

    def _best_orientations_for_order(
        self, start: int, end: int, edge_indices: tuple[int, ...], order: Iterable[int]
    ) -> tuple[RouteStep, ...]:
        sequence = tuple(order)
        dynamic: list[list[float]] = [[float("inf"), float("inf")] for _ in sequence]
        parent: list[list[int | None]] = [[None, None] for _ in sequence]

        def endpoints(local_edge: int, orientation: int) -> tuple[int, int]:
            edge = self.instance.required_edges[edge_indices[local_edge]]
            return (edge.u, edge.v) if orientation == 0 else (edge.v, edge.u)

        first_edge = self.instance.required_edges[edge_indices[sequence[0]]]
        for orientation in (0, 1):
            edge_start, _ = endpoints(sequence[0], orientation)
            dynamic[0][orientation] = self.instance.cost(start, edge_start) + first_edge.service_cost
        for position in range(1, len(sequence)):
            edge = self.instance.required_edges[edge_indices[sequence[position]]]
            for orientation in (0, 1):
                edge_start, _ = endpoints(sequence[position], orientation)
                for previous_orientation in (0, 1):
                    _, previous_end = endpoints(sequence[position - 1], previous_orientation)
                    trial = dynamic[position - 1][previous_orientation] + self.instance.cost(previous_end, edge_start) + edge.service_cost
                    if trial + EPSILON < dynamic[position][orientation]:
                        dynamic[position][orientation] = trial
                        parent[position][orientation] = previous_orientation
        final_orientation = min(
            (0, 1),
            key=lambda orientation: dynamic[-1][orientation] + self.instance.cost(endpoints(sequence[-1], orientation)[1], end),
        )
        orientations: list[tuple[int, int]] = []
        orientation: int | None = final_orientation
        for position in range(len(sequence) - 1, -1, -1):
            assert orientation is not None
            orientations.append((sequence[position], orientation))
            orientation = parent[position][orientation]
        orientations.reverse()
        return self._steps_from_order(start, end, edge_indices, orientations)

    def _steps_from_order(
        self,
        start: int,
        end: int,
        edge_indices: tuple[int, ...],
        order_orientations: Iterable[tuple[int, int]],
    ) -> tuple[RouteStep, ...]:
        current = start
        steps: list[RouteStep] = []
        for local_edge, orientation in order_orientations:
            edge_index = edge_indices[local_edge]
            edge = self.instance.required_edges[edge_index]
            service_start, service_end = (edge.u, edge.v) if orientation == 0 else (edge.v, edge.u)
            if current != service_start:
                steps.append(RouteStep("deadhead", current, service_start, self.instance.cost(current, service_start)))
            steps.append(RouteStep("service", service_start, service_end, edge.service_cost, edge_index))
            current = service_end
        if current != end:
            steps.append(RouteStep("deadhead", current, end, self.instance.cost(current, end)))
        if not steps:  # Defensive: edge-less stages are handled by _decode_stage.
            steps.append(RouteStep("deadhead", start, end, self.instance.cost(start, end)))
        return tuple(steps)

    def _stage_multipliers(self, order: tuple[int, ...]) -> tuple[float, ...]:
        load = self.instance.total_demand
        values: list[float] = []
        for stage in range(self.instance.stage_count):
            values.append(self.instance.curb_weight + load)
            if stage < len(order):
                load -= self.instance.delivery_demands[order[stage]]
        return tuple(values)

    def _delivery_order_dp(self, candidate: Candidate) -> Candidate:
        """Exactly optimize delivery order while fixing all edge-stage labels.

        For a fixed edge allocation, stage ``p`` has a known set of required
        edges.  The only remaining coupled decision is which delivery ends
        each stage; its multiplier depends on the delivered-node subset.  The
        subset dynamic program solves that restricted problem in
        O(2^k k^2), where k is the number of deliveries.  Stage routes are
        decoded through the usual cache, so this is an exact ordering solve
        relative to the currently selected stage decoder.
        """
        deliveries = tuple(candidate.delivery_order)
        count = len(deliveries)
        if count < 2:
            return candidate
        groups: list[list[int]] = [[] for _ in range(self.instance.stage_count)]
        for edge_index, stage in enumerate(candidate.edge_stages):
            groups[stage].append(edge_index)
        group_tuples = tuple(tuple(group) for group in groups)
        demands = tuple(self.instance.delivery_demands[node] for node in deliveries)
        subset_demand = [0.0] * (1 << count)
        for mask in range(1, 1 << count):
            low_bit = mask & -mask
            index = low_bit.bit_length() - 1
            subset_demand[mask] = subset_demand[mask ^ low_bit] + demands[index]

        route_cost_cache: dict[tuple[int, int, int], float] = {}

        def stage_cost(stage: int, start: int, end: int, delivered_mask: int) -> float:
            key = (stage, start, end)
            base = route_cost_cache.get(key)
            if base is None:
                base = self._stage_base_cost(start, end, group_tuples[stage])
                route_cost_cache[key] = base
            multiplier = self.instance.curb_weight + self.instance.total_demand - subset_demand[delivered_mask]
            return multiplier * (base + self.instance.landing_takeoff_cost)

        costs: dict[tuple[int, int], float] = {}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {}
        for index, node in enumerate(deliveries):
            if self._deadline_reached():
                return candidate
            mask = 1 << index
            costs[(mask, index)] = stage_cost(0, self.instance.depot, node, 0)
            parent[(mask, index)] = None

        full_mask = (1 << count) - 1
        for mask in range(1, full_mask + 1):
            delivered_count = mask.bit_count()
            if delivered_count == count:
                continue
            for last in range(count):
                key = (mask, last)
                current_cost = costs.get(key)
                if current_cost is None:
                    continue
                if self._deadline_reached():
                    return candidate
                for nxt in range(count):
                    if mask & (1 << nxt):
                        continue
                    next_mask = mask | (1 << nxt)
                    trial = current_cost + stage_cost(
                        delivered_count, deliveries[last], deliveries[nxt], mask
                    )
                    next_key = (next_mask, nxt)
                    if trial + EPSILON < costs.get(next_key, float("inf")):
                        costs[next_key] = trial
                        parent[next_key] = key

        final_key = min(
            ((full_mask, last) for last in range(count)),
            key=lambda key: costs[key]
            + stage_cost(count, deliveries[key[1]], self.instance.depot, full_mask),
        )
        order_indices: list[int] = []
        key: tuple[int, int] | None = final_key
        while key is not None:
            order_indices.append(key[1])
            key = parent[key]
        order_indices.reverse()
        return Candidate(tuple(deliveries[index] for index in order_indices), candidate.edge_stages)

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        total = sum(weights.values())
        pivot = self.rng.random() * total
        accumulator = 0.0
        for name, weight in weights.items():
            accumulator += weight
            if accumulator >= pivot:
                return name
        return next(reversed(weights))

    def _delivery_relocate(self, candidate: Candidate) -> Candidate:
        order = list(candidate.delivery_order)
        if len(order) < 2:
            return candidate
        source = self.rng.randrange(len(order))
        node = order.pop(source)
        target = self.rng.randrange(len(order) + 1)
        order.insert(target, node)
        return Candidate(tuple(order), candidate.edge_stages)

    def _delivery_swap(self, candidate: Candidate) -> Candidate:
        order = list(candidate.delivery_order)
        if len(order) < 2:
            return candidate
        first, second = self.rng.sample(range(len(order)), 2)
        order[first], order[second] = order[second], order[first]
        return Candidate(tuple(order), candidate.edge_stages)

    def _delivery_reverse(self, candidate: Candidate) -> Candidate:
        order = list(candidate.delivery_order)
        if len(order) < 2:
            return candidate
        first, second = sorted(self.rng.sample(range(len(order)), 2))
        order[first : second + 1] = reversed(order[first : second + 1])
        return Candidate(tuple(order), candidate.edge_stages)

    def _delivery_ruin_recreate(self, candidate: Candidate) -> Candidate:
        """Remove up to two deliveries and exactly choose their reinsertion slots.

        ``evaluate`` deliberately rejects partial delivery orders, because their
        number of stages and remaining loads are undefined.  Consequently this
        operator enumerates complete reinsertion combinations before scoring
        them rather than attempting to greedily score an invalid partial order.
        """
        order = list(candidate.delivery_order)
        if len(order) < 2:
            return candidate
        remove_count = min(len(order) - 1, self.rng.randint(1, 2))
        removed = [order.pop(self.rng.randrange(len(order))) for _ in range(remove_count)]
        completed_orders: list[tuple[int, ...]] = []
        for reinsertion_order in permutations(removed):
            partial_orders = [tuple(order)]
            for node in reinsertion_order:
                partial_orders = [
                    partial[:position] + (node,) + partial[position:]
                    for partial in partial_orders
                    for position in range(len(partial) + 1)
                ]
            completed_orders.extend(partial_orders)
        best = candidate
        best_objective = float("inf")
        for order in completed_orders:
            if self._deadline_reached():
                break
            proposal = Candidate(order, candidate.edge_stages)
            objective = self.evaluate(proposal).objective
            if objective + EPSILON < best_objective:
                best, best_objective = proposal, objective
        return best

    def _edge_reassign(self, candidate: Candidate) -> Candidate:
        if not candidate.edge_stages or self.instance.stage_count < 2:
            return candidate
        stages = list(candidate.edge_stages)
        edge = self.rng.randrange(len(stages))
        alternatives = [stage for stage in range(self.instance.stage_count) if stage != stages[edge]]
        stages[edge] = self.rng.choice(alternatives)
        return Candidate(candidate.delivery_order, tuple(stages))

    def _edge_best_reassign(self, candidate: Candidate) -> Candidate:
        """Choose one edge, then exactly select its best stage assignment.

        This is a small but useful intensification move: with delivery order and
        all other edge labels fixed, it exhaustively considers every allowed
        stage for the selected edge.  It is bounded by the number of deliveries
        rather than by the number of required edges.
        """
        if not candidate.edge_stages or self.instance.stage_count < 2:
            return candidate
        edge_index = self.rng.randrange(len(candidate.edge_stages))
        best = candidate
        best_evaluation = self.evaluate(candidate)
        for stage in range(self.instance.stage_count):
            if stage == candidate.edge_stages[edge_index] or self._deadline_reached():
                continue
            stages = list(candidate.edge_stages)
            stages[edge_index] = stage
            proposal = Candidate(candidate.delivery_order, tuple(stages))
            evaluation = self.evaluate(proposal)
            if evaluation.objective + EPSILON < best_evaluation.objective:
                best, best_evaluation = proposal, evaluation
        return best

    def _edge_cluster_reassign(self, candidate: Candidate) -> Candidate:
        if len(candidate.edge_stages) < 2 or self.instance.stage_count < 2:
            return candidate
        seed = self.rng.randrange(len(candidate.edge_stages))
        seed_edge = self.instance.required_edges[seed]

        def relatedness(index: int) -> float:
            edge = self.instance.required_edges[index]
            return min(
                self.instance.cost(seed_edge.u, edge.u),
                self.instance.cost(seed_edge.u, edge.v),
                self.instance.cost(seed_edge.v, edge.u),
                self.instance.cost(seed_edge.v, edge.v),
            )

        cluster_size = self.rng.randint(2, min(5, len(candidate.edge_stages)))
        cluster = sorted(range(len(candidate.edge_stages)), key=relatedness)[:cluster_size]
        stages = list(candidate.edge_stages)
        destination = self.rng.randrange(self.instance.stage_count)
        for edge in cluster:
            stages[edge] = destination
        return Candidate(candidate.delivery_order, tuple(stages))

    def _edge_cluster_recreate(self, candidate: Candidate) -> Candidate:
        """Destroy a related edge cluster and greedily reinsert it by true cost.

        The delivery order is fixed, but the reconstruction evaluates route
        sharing within each stage rather than treating each edge independently.
        It therefore complements the bounded exact two-stage repair with a
        wider, approximate multi-stage neighbourhood.
        """
        if len(candidate.edge_stages) < 2:
            return candidate
        seed_index = self.rng.randrange(len(candidate.edge_stages))
        seed_edge = self.instance.required_edges[seed_index]

        def relatedness(index: int) -> float:
            edge = self.instance.required_edges[index]
            return min(
                self.instance.cost(seed_edge.u, edge.u),
                self.instance.cost(seed_edge.u, edge.v),
                self.instance.cost(seed_edge.v, edge.u),
                self.instance.cost(seed_edge.v, edge.v),
            )

        cluster_size = self.rng.randint(2, min(6, len(candidate.edge_stages)))
        removed = tuple(sorted(range(len(candidate.edge_stages)), key=relatedness)[:cluster_size])
        removed_set = set(removed)
        groups: list[list[int]] = [[] for _ in range(self.instance.stage_count)]
        for edge_index, stage in enumerate(candidate.edge_stages):
            if edge_index not in removed_set:
                groups[stage].append(edge_index)

        starts = (self.instance.depot, *candidate.delivery_order)
        ends = (*candidate.delivery_order, self.instance.depot)
        multipliers = self._stage_multipliers(candidate.delivery_order)
        base_costs = [
            self._stage_base_cost(starts[stage], ends[stage], tuple(groups[stage]))
            for stage in range(self.instance.stage_count)
        ]
        new_stages = list(candidate.edge_stages)
        # Higher service-cost edges have the larger load-stage consequence; put
        # them first so later insertions can exploit their established routes.
        for edge_index in sorted(
            removed, key=lambda index: self.instance.required_edges[index].service_cost, reverse=True
        ):
            if self._deadline_reached():
                chosen = min(range(self.instance.stage_count), key=lambda stage: multipliers[stage])
                groups[chosen].append(edge_index)
                new_stages[edge_index] = chosen
                continue
            trial_costs = [
                self._stage_base_cost(
                    starts[stage], ends[stage], tuple((*groups[stage], edge_index))
                )
                for stage in range(self.instance.stage_count)
            ]
            chosen = min(
                range(self.instance.stage_count),
                key=lambda stage: multipliers[stage] * (trial_costs[stage] - base_costs[stage]),
            )
            groups[chosen].append(edge_index)
            new_stages[edge_index] = chosen
            base_costs[chosen] = trial_costs[chosen]
        return Candidate(candidate.delivery_order, tuple(new_stages))

    def _load_aware_ruin_recreate(self, candidate: Candidate) -> Candidate:
        """Destroy costly service edges and reinsert them by weighted regret.

        A required edge is not intrinsically expensive in LDdGRP: its impact
        depends on the remaining-load multiplier of the stage that services it
        and on the detour it creates in that stage's decoded route.  This
        operator therefore removes edges with the largest *fixed-route*,
        load-weighted removal saving.  It then rebuilds the destroyed part by
        regret insertion using the actual decoder for every considered stage.

        The fixed-route removal scores are only a cheap selection heuristic;
        all insertions and the final acceptance are evaluated with the normal
        LDdGRP objective.  The method is consequently a valid heuristic
        neighbourhood, not a delta formula claimed to be exact for the
        large-stage decoder.
        """
        if not candidate.edge_stages:
            return candidate
        incumbent = self.evaluate(candidate)
        global_deadline = self._search_deadline if self._search_deadline is not None else float("inf")
        deadline = min(global_deadline, monotonic() + self.config.load_aware_ruin_seconds)
        multipliers = self._stage_multipliers(candidate.delivery_order)
        scored: list[tuple[float, int]] = []

        for route in incumbent.stages:
            services = [step for step in route.steps if step.kind == "service"]
            for position, step in enumerate(services):
                # The route is contiguous, so these are precisely the two
                # nodes that would be joined if the service action disappeared
                # while all other service actions and directions stayed fixed.
                previous = route.start_node if position == 0 else services[position - 1].to_node
                following = route.end_node if position + 1 == len(services) else services[position + 1].from_node
                saving = (
                    self.instance.cost(previous, step.from_node)
                    + step.cost
                    + self.instance.cost(step.to_node, following)
                    - self.instance.cost(previous, following)
                )
                if step.edge_index is None:
                    raise ValueError("A service step lacks its required-edge index.")
                scored.append((multipliers[route.stage] * saving, step.edge_index))

        if not scored:
            return candidate
        # The edge count is deliberately small: every regret choice evaluates
        # multiple decoder calls, and this operator is intended for tight
        # wall-clock budgets.
        remove_count = min(len(scored), self.config.load_aware_ruin_max_edges)
        removed = tuple(edge_index for _, edge_index in sorted(scored, reverse=True)[:remove_count])
        removed_set = set(removed)
        groups: list[list[int]] = [[] for _ in range(self.instance.stage_count)]
        for edge_index, stage in enumerate(candidate.edge_stages):
            if edge_index not in removed_set:
                groups[stage].append(edge_index)

        starts = (self.instance.depot, *candidate.delivery_order)
        ends = (*candidate.delivery_order, self.instance.depot)
        base_costs = [
            self._stage_base_cost(starts[stage], ends[stage], tuple(groups[stage]))
            for stage in range(self.instance.stage_count)
        ]
        labels = list(candidate.edge_stages)
        pending = set(removed)
        while pending and monotonic() < deadline and not self._deadline_reached():
            choices: list[tuple[float, float, int, int, float]] = []
            # A choice is (regret, best weighted marginal, edge, stage,
            # trial base cost).  High regret protects an edge whose cheap
            # placement may vanish if other edges are placed first.
            for edge_index in pending:
                options: list[tuple[float, int, float]] = []
                for stage in range(self.instance.stage_count):
                    if monotonic() >= deadline or self._deadline_reached():
                        break
                    trial_base = self._stage_base_cost(
                        starts[stage], ends[stage], tuple((*groups[stage], edge_index))
                    )
                    marginal = multipliers[stage] * (trial_base - base_costs[stage])
                    options.append((marginal, stage, trial_base))
                if len(options) != self.instance.stage_count:
                    break
                options.sort()
                best_marginal, best_stage, best_base = options[0]
                second_marginal = options[1][0] if len(options) > 1 else best_marginal
                choices.append((second_marginal - best_marginal, best_marginal, edge_index, best_stage, best_base))
            if not choices:
                break
            # Prioritise regret; the second key makes equal-regret selections
            # favour the least costly insertion.
            _, _, chosen_edge, chosen_stage, chosen_base = max(
                choices, key=lambda choice: (choice[0], -choice[1])
            )
            groups[chosen_stage].append(chosen_edge)
            labels[chosen_edge] = chosen_stage
            base_costs[chosen_stage] = chosen_base
            pending.remove(chosen_edge)

        # A partial rebuild is not useful under a tight deadline: restore all
        # unprocessed edges to their original stages before scoring the result.
        for edge_index in pending:
            labels[edge_index] = candidate.edge_stages[edge_index]
        proposal = Candidate(candidate.delivery_order, tuple(labels))
        proposal_evaluation = self.evaluate(proposal)
        return proposal if proposal_evaluation.objective + EPSILON < incumbent.objective else candidate

    def _route_block_relocate(self, candidate: Candidate) -> Candidate:
        """Relocate a costly contiguous service block to its best other stage.

        This is a route-cutting neighbourhood inspired by arc-routing
        decomposition methods.  The candidate representation deliberately
        stores stage labels rather than route order, hence a block move changes
        the selected edges' stage labels and lets the standard decoder choose
        their insertion positions and orientations.  It is retained only after
        that full re-decoding proves a strict improvement.
        """
        if self.instance.stage_count < 2 or len(candidate.edge_stages) < 2:
            return candidate
        incumbent = self.evaluate(candidate)
        global_deadline = self._search_deadline if self._search_deadline is not None else float("inf")
        deadline = min(global_deadline, monotonic() + self.config.route_block_seconds)
        multipliers = self._stage_multipliers(candidate.delivery_order)
        block_options: list[tuple[float, tuple[int, ...], int]] = []

        for route in incumbent.stages:
            services = [step for step in route.steps if step.kind == "service"]
            maximum = min(self.config.route_block_max_edges, len(services))
            for size in range(1, maximum + 1):
                for first in range(len(services) - size + 1):
                    last = first + size - 1
                    previous = route.start_node if first == 0 else services[first - 1].to_node
                    following = route.end_node if last + 1 == len(services) else services[last + 1].from_node
                    retained = self.instance.cost(previous, following)
                    removed_cost = self.instance.cost(previous, services[first].from_node)
                    for position in range(first, last + 1):
                        step = services[position]
                        removed_cost += step.cost
                        if position < last:
                            removed_cost += self.instance.cost(step.to_node, services[position + 1].from_node)
                    removed_cost += self.instance.cost(services[last].to_node, following)
                    block = tuple(step.edge_index for step in services[first : last + 1])
                    if any(edge_index is None for edge_index in block):
                        raise ValueError("A service step lacks its required-edge index.")
                    block_options.append(
                        (
                            multipliers[route.stage] * (removed_cost - retained),
                            tuple(int(edge_index) for edge_index in block),
                            route.stage,
                        )
                    )

        if not block_options:
            return candidate
        # Scanning one promising block across all target stages is much cheaper
        # than a full cross-product of source blocks and destinations.
        _, block, source_stage = max(block_options, key=lambda option: option[0])
        best = incumbent
        for target_stage in range(self.instance.stage_count):
            if target_stage == source_stage or monotonic() >= deadline or self._deadline_reached():
                continue
            labels = list(candidate.edge_stages)
            for edge_index in block:
                labels[edge_index] = target_stage
            proposal = Candidate(candidate.delivery_order, tuple(labels))
            evaluation = self.evaluate(proposal)
            if evaluation.objective + EPSILON < best.objective:
                best = evaluation
        return best.candidate

    def _cache_discovered_stage_route(
        self, start: int, end: int, edge_indices: tuple[int, ...], route: tuple[RouteStep, ...]
    ) -> None:
        """Keep a newly discovered feasible route only when it improves decoding.

        A candidate records edge-to-stage labels, not its within-stage service
        permutation.  A sequence-split move nevertheless discovers a concrete
        feasible permutation.  Retaining it in the ordinary stage cache means
        a subsequent evaluation of that same edge set cannot be worse merely
        because its generic greedy decoder starts from a different sequence.
        """
        key = (start, end, self._active_large_stage_local_search_passes, tuple(sorted(edge_indices)))
        existing = self._decode_stage(start, end, edge_indices)
        if sum(step.cost for step in route) + EPSILON < sum(step.cost for step in existing):
            self._stage_cache[key] = route

    def _split_service_sequence(
        self, delivery_order: tuple[int, ...], sequence: tuple[int, ...], deadline: float
    ) -> Candidate | None:
        """Optimally split one ordered service sequence across delivery stages.

        The split restriction assigns every stage a contiguous subsequence of
        ``sequence`` (empty stages are allowed).  Given that restriction and a
        fixed delivery order, the dynamic programme exactly chooses all stage
        boundaries and every service direction.  It is not an exact LDdGRP
        algorithm because arbitrary stage allocations need not be contiguous
        in a single global service sequence.
        """
        edge_count = len(sequence)
        stage_count = self.instance.stage_count
        starts = (self.instance.depot, *delivery_order)
        ends = (*delivery_order, self.instance.depot)
        multipliers = self._stage_multipliers(delivery_order)
        # ``segment_costs[stage][first][last]`` is the least base cost of
        # traversing ordered services sequence[first:last] between that
        # stage's endpoints.  A two-state orientation recursion gives every
        # interval in O(m^2) for each stage.
        segment_costs: list[list[list[float]]] = []
        for stage in range(stage_count):
            costs = [[float("inf")] * (edge_count + 1) for _ in range(edge_count + 1)]
            for first in range(edge_count + 1):
                costs[first][first] = self.instance.cost(starts[stage], ends[stage])
                if first == edge_count:
                    continue
                edge = self.instance.required_edges[sequence[first]]
                forward = self.instance.cost(starts[stage], edge.u) + edge.service_cost
                backward = self.instance.cost(starts[stage], edge.v) + edge.service_cost
                # Values are the best path cost ending after the last service
                # in each orientation: 0 means u->v, 1 means v->u.
                orientation_cost = [forward, backward]
                for last in range(first, edge_count):
                    if last != first:
                        previous = self.instance.required_edges[sequence[last - 1]]
                        current = self.instance.required_edges[sequence[last]]
                        next_cost = [float("inf"), float("inf")]
                        endpoints = ((current.u, current.v), (current.v, current.u))
                        previous_ends = (previous.v, previous.u)
                        for orientation, (service_start, _) in enumerate(endpoints):
                            next_cost[orientation] = min(
                                orientation_cost[previous_orientation]
                                + self.instance.cost(previous_ends[previous_orientation], service_start)
                                + current.service_cost
                                for previous_orientation in (0, 1)
                            )
                        orientation_cost = next_cost
                    current = self.instance.required_edges[sequence[last]]
                    costs[first][last + 1] = min(
                        orientation_cost[0] + self.instance.cost(current.v, ends[stage]),
                        orientation_cost[1] + self.instance.cost(current.u, ends[stage]),
                    )
                    if last % 16 == 0 and monotonic() >= deadline:
                        return None
            segment_costs.append(costs)

        previous = [float("inf")] * (edge_count + 1)
        previous[0] = 0.0
        parents: list[list[int | None]] = []
        for stage in range(stage_count):
            current = [float("inf")] * (edge_count + 1)
            parent: list[int | None] = [None] * (edge_count + 1)
            for last in range(edge_count + 1):
                for first in range(last + 1):
                    if previous[first] == float("inf"):
                        continue
                    trial = previous[first] + multipliers[stage] * (
                        segment_costs[stage][first][last] + self.instance.landing_takeoff_cost
                    )
                    if trial + EPSILON < current[last]:
                        current[last] = trial
                        parent[last] = first
                if last % 16 == 0 and monotonic() >= deadline:
                    return None
            parents.append(parent)
            previous = current
        if previous[edge_count] == float("inf"):
            return None

        boundaries: list[tuple[int, int]] = [(0, 0)] * stage_count
        last = edge_count
        for stage in range(stage_count - 1, -1, -1):
            first = parents[stage][last]
            if first is None:
                raise ValueError("Sequence split dynamic programme has no predecessor.")
            boundaries[stage] = (first, last)
            last = first
        if last != 0:
            raise ValueError("Sequence split dynamic programme did not cover the full sequence.")

        labels = [0] * len(self.instance.required_edges)
        for stage, (first, last) in enumerate(boundaries):
            stage_edges = sequence[first:last]
            for edge_index in stage_edges:
                labels[edge_index] = stage
            if stage_edges:
                route = self._best_orientations_for_order(
                    starts[stage], ends[stage], stage_edges, range(len(stage_edges))
                )
                self._cache_discovered_stage_route(starts[stage], ends[stage], stage_edges, route)
        return Candidate(delivery_order, tuple(labels))

    def _perturb_service_sequence(self, sequence: tuple[int, ...]) -> tuple[int, ...]:
        """Apply one small VNS move to a global required-edge sequence."""
        if len(sequence) < 2:
            return sequence
        move = self.rng.randrange(4)
        if move == 0:  # 2-exchange
            first, second = self.rng.sample(range(len(sequence)), 2)
            values = list(sequence)
            values[first], values[second] = values[second], values[first]
            return tuple(values)
        if move == 1:  # 1-opt relocation
            values = list(sequence)
            source = self.rng.randrange(len(values))
            edge = values.pop(source)
            values.insert(self.rng.randrange(len(values) + 1), edge)
            return tuple(values)
        if move == 2:  # 2-opt reversal
            first, second = sorted(self.rng.sample(range(len(sequence)), 2))
            return sequence[:first] + tuple(reversed(sequence[first : second + 1])) + sequence[second + 1 :]
        # Short-chain relocation, analogous to an arc-routing block exchange.
        maximum = min(4, len(sequence))
        size = self.rng.randint(2, maximum)
        first = self.rng.randrange(len(sequence) - size + 1)
        block = sequence[first : first + size]
        remaining = sequence[:first] + sequence[first + size :]
        target = self.rng.randrange(len(remaining) + 1)
        return remaining[:target] + block + remaining[target:]

    def _sequence_split_vnd(self, candidate: Candidate) -> Candidate:
        """Use a restricted sequence/split VND as a short exact-decoded repair.

        It begins with the current decoded stages flattened in stage order,
        optimally repartitions that sequence, then tests a bounded number of
        swap/relocate/reverse/block perturbations.  Standard LDdGRP evaluation
        remains the acceptance criterion, so the restricted split objective
        can never replace the actual objective by accident.
        """
        incumbent = self.evaluate(candidate)
        sequence = tuple(
            step.edge_index
            for route in incumbent.stages
            for step in route.steps
            if step.kind == "service" and step.edge_index is not None
        )
        if len(sequence) != len(candidate.edge_stages) or len(set(sequence)) != len(sequence):
            raise ValueError("Decoded stages do not form a required-edge permutation.")
        global_deadline = self._search_deadline if self._search_deadline is not None else float("inf")
        deadline = min(global_deadline, monotonic() + self.config.sequence_split_seconds)
        best = incumbent
        current_sequence = sequence
        split = self._split_service_sequence(candidate.delivery_order, current_sequence, deadline)
        if split is None:
            return candidate
        split_evaluation = self.evaluate(split)
        if split_evaluation.objective + EPSILON < best.objective:
            best = split_evaluation

        for _ in range(self.config.sequence_split_max_moves):
            if monotonic() >= deadline or self._deadline_reached():
                break
            trial_sequence = self._perturb_service_sequence(current_sequence)
            if trial_sequence == current_sequence:
                continue
            proposal = self._split_service_sequence(candidate.delivery_order, trial_sequence, deadline)
            if proposal is None:
                break
            proposal_evaluation = self.evaluate(proposal)
            if proposal_evaluation.objective + EPSILON < best.objective:
                best = proposal_evaluation
                current_sequence = trial_sequence
        return best.candidate

    def _edge_cluster_stage_descent(self, candidate: Candidate) -> Candidate:
        """Reassign a related edge cluster by the true decoded objective.

        Earlier reconstruction moves use a stage-wise marginal-cost surrogate.
        This operator instead evaluates every tested edge-to-stage relocation as
        a complete candidate, retaining only strict improvements.  The cluster,
        number of passes, and clock budget bound its cost; unchanged stage
        routes are retrieved from the decoder cache.
        """
        if not candidate.edge_stages or self.instance.stage_count < 2:
            return candidate
        seed_index = self.rng.randrange(len(candidate.edge_stages))
        seed_edge = self.instance.required_edges[seed_index]

        def relatedness(index: int) -> float:
            edge = self.instance.required_edges[index]
            return min(
                self.instance.cost(seed_edge.u, edge.u),
                self.instance.cost(seed_edge.u, edge.v),
                self.instance.cost(seed_edge.v, edge.u),
                self.instance.cost(seed_edge.v, edge.v),
            )

        cluster = tuple(
            sorted(range(len(candidate.edge_stages)), key=relatedness)[: self.config.stage_descent_cluster_size]
        )
        deadline = min(
            self._search_deadline if self._search_deadline is not None else float("inf"),
            monotonic() + self.config.stage_descent_seconds,
        )
        current = candidate
        current_evaluation = self.evaluate(current)
        for _ in range(self.config.stage_descent_passes):
            moved = False
            # Random order provides diversification, while spatial clustering
            # retains the route-sharing structure that single-edge moves miss.
            edge_order = list(cluster)
            self.rng.shuffle(edge_order)
            for edge_index in edge_order:
                if monotonic() >= deadline or self._deadline_reached():
                    return current
                best_candidate = current
                best_evaluation = current_evaluation
                stage_order = list(range(self.instance.stage_count))
                self.rng.shuffle(stage_order)
                for stage in stage_order:
                    if stage == current.edge_stages[edge_index]:
                        continue
                    if monotonic() >= deadline or self._deadline_reached():
                        return current
                    stages = list(current.edge_stages)
                    stages[edge_index] = stage
                    proposal = Candidate(current.delivery_order, tuple(stages))
                    proposal_evaluation = self.evaluate(proposal)
                    if proposal_evaluation.objective + EPSILON < best_evaluation.objective:
                        best_candidate, best_evaluation = proposal, proposal_evaluation
                if best_candidate != current:
                    current, current_evaluation = best_candidate, best_evaluation
                    moved = True
            if not moved:
                break
        return current

    def _spatial_two_stage_exact_repair(self, candidate: Candidate) -> Candidate:
        """Exactly repartition a spatial edge cluster between two load stages.

        This is an exact solve of the specified binary assignment neighbourhood,
        not a global LDdGRP solve.  Unlike the earlier adjacent-stage repair,
        its destroyed edges are selected by geography even when they currently
        belong to several different stages.  That exposes the route-sharing
        moves the independent edge operators cannot make.
        """
        if not candidate.edge_stages or self.instance.stage_count < 2:
            return candidate
        seed_index = self.rng.randrange(len(candidate.edge_stages))
        seed_edge = self.instance.required_edges[seed_index]

        def relatedness(index: int) -> float:
            edge = self.instance.required_edges[index]
            return min(
                self.instance.cost(seed_edge.u, edge.u),
                self.instance.cost(seed_edge.u, edge.v),
                self.instance.cost(seed_edge.v, edge.u),
                self.instance.cost(seed_edge.v, edge.v),
            )

        mutable = tuple(
            sorted(range(len(candidate.edge_stages)), key=relatedness)[: self.config.spatial_repair_max_edges]
        )
        first_stage = candidate.edge_stages[seed_index]
        alternatives = [stage for stage in range(self.instance.stage_count) if stage != first_stage]
        adjacent = [stage for stage in alternatives if abs(stage - first_stage) == 1]
        if adjacent and self.rng.random() < 0.65:
            second_stage = self.rng.choice(adjacent)
        else:
            second_stage = self.rng.choice(alternatives)
        deadline = min(
            self._search_deadline if self._search_deadline is not None else float("inf"),
            monotonic() + self.config.spatial_repair_seconds,
        )
        return self.repair_oracle.repair(
            candidate,
            self.evaluate,
            mutable_edge_indices=mutable,
            allowed_stages=(first_stage, second_stage),
            deadline=deadline,
        )

    def _frozen_skeleton_stage_subset_costs(
        self,
        start: int,
        end: int,
        frozen_services: tuple[tuple[int, int, int], ...],
        mutable_edge_indices: tuple[int, ...],
        deadline: float,
    ) -> tuple[float, ...] | None:
        """Return the exact cost of each mutable-edge subset on a frozen skeleton.

        ``frozen_services`` gives the required-edge service order and direction
        that remain unchanged in a stage.  Every subset of the mutable edges is
        then inserted at its best positions and orientations.  This is exact
        for that skeleton neighbourhood: the dynamic program preserves the
        frozen service sequence, but otherwise chooses the mutable edge order
        and orientation freely.
        """
        mutable_count = len(mutable_edge_indices)
        subset_count = 1 << mutable_count
        fixed_count = len(frozen_services)
        # An action either services the next frozen edge or a previously
        # unserved mutable edge.  Thus every transition advances exactly one
        # layer, making the label-setting dynamic program acyclic.
        layers: list[dict[tuple[int, int, int], float]] = [
            {} for _ in range(fixed_count + mutable_count + 1)
        ]
        layers[0][(0, 0, start)] = 0.0

        def relax(
            target: dict[tuple[int, int, int], float], state: tuple[int, int, int], value: float
        ) -> None:
            if value + EPSILON < target.get(state, float("inf")):
                target[state] = value

        for progress, layer in enumerate(layers[:-1]):
            for state_number, ((fixed_seen, mutable_mask, current), value) in enumerate(layer.items()):
                if state_number % 64 == 0 and monotonic() >= deadline:
                    return None
                next_layer = layers[progress + 1]
                if fixed_seen < fixed_count:
                    edge_index, service_start, service_end = frozen_services[fixed_seen]
                    edge = self.instance.required_edges[edge_index]
                    relax(
                        next_layer,
                        (fixed_seen + 1, mutable_mask, service_end),
                        value + self.instance.cost(current, service_start) + edge.service_cost,
                    )
                for local_edge, edge_index in enumerate(mutable_edge_indices):
                    if mutable_mask & (1 << local_edge):
                        continue
                    edge = self.instance.required_edges[edge_index]
                    next_mask = mutable_mask | (1 << local_edge)
                    for service_start, service_end in ((edge.u, edge.v), (edge.v, edge.u)):
                        relax(
                            next_layer,
                            (fixed_seen, next_mask, service_end),
                            value + self.instance.cost(current, service_start) + edge.service_cost,
                        )

        costs = [float("inf")] * subset_count
        for layer in layers:
            for (fixed_seen, mutable_mask, current), value in layer.items():
                if fixed_seen == fixed_count:
                    costs[mutable_mask] = min(costs[mutable_mask], value + self.instance.cost(current, end))
        return tuple(costs)

    def _skeleton_two_stage_exact_repair(self, candidate: Candidate) -> Candidate:
        """Exactly reallocate a cluster while retaining the surrounding route skeleton.

        The ordinary enumerative repair must fully decode both stages for every
        binary assignment.  Here the service order and orientation of all
        non-destroyed edges are fixed from the incumbent certificate.  A dynamic
        program computes the best insertion route for *every* subset of the
        destroyed cluster in each target stage, then enumerates the two-stage
        split.  It is a materially larger exact neighbourhood at a bounded
        cost, while still not a global LDdGRP exact solve.
        """
        if not candidate.edge_stages or self.instance.stage_count < 2:
            return candidate
        seed_index = self.rng.randrange(len(candidate.edge_stages))
        first_stage = candidate.edge_stages[seed_index]
        alternatives = [stage for stage in range(self.instance.stage_count) if stage != first_stage]
        adjacent = [stage for stage in alternatives if abs(stage - first_stage) == 1]
        second_stage = self.rng.choice(adjacent) if adjacent and self.rng.random() < 0.65 else self.rng.choice(alternatives)
        allowed_stages = (first_stage, second_stage)
        seed_edge = self.instance.required_edges[seed_index]

        def relatedness(index: int) -> float:
            edge = self.instance.required_edges[index]
            return min(
                self.instance.cost(seed_edge.u, edge.u),
                self.instance.cost(seed_edge.u, edge.v),
                self.instance.cost(seed_edge.v, edge.u),
                self.instance.cost(seed_edge.v, edge.v),
            )

        eligible = [
            index for index, stage in enumerate(candidate.edge_stages) if stage in allowed_stages
        ]
        mutable = tuple(sorted(eligible, key=relatedness)[: self.config.skeleton_repair_max_edges])
        if len(mutable) < 2:
            return candidate
        deadline = min(
            self._search_deadline if self._search_deadline is not None else float("inf"),
            monotonic() + self.config.skeleton_repair_seconds,
        )
        mutable_set = set(mutable)
        evaluation = self.evaluate(candidate)
        profiles: list[tuple[float, ...]] = []
        for stage in allowed_stages:
            route = evaluation.stages[stage]
            frozen_services = tuple(
                (step.edge_index, step.from_node, step.to_node)
                for step in route.steps
                if step.kind == "service" and step.edge_index not in mutable_set
            )
            # ``RouteStep.edge_index`` is present for every service step, as
            # independently checked by validate_evaluation.
            if any(edge_index is None for edge_index, _, _ in frozen_services):
                raise ValueError("A stage service step lacks its required-edge index.")
            profile = self._frozen_skeleton_stage_subset_costs(
                route.start_node,
                route.end_node,
                tuple((int(edge_index), service_start, service_end) for edge_index, service_start, service_end in frozen_services),
                mutable,
                deadline,
            )
            if profile is None:
                return candidate
            profiles.append(profile)
        if monotonic() >= deadline:
            return candidate

        multipliers = self._stage_multipliers(candidate.delivery_order)
        first_multiplier, second_multiplier = (multipliers[stage] for stage in allowed_stages)
        full_mask = (1 << len(mutable)) - 1
        best_mask = 0
        best_cost = float("inf")
        for first_mask in range(full_mask + 1):
            second_mask = full_mask ^ first_mask
            trial = first_multiplier * (profiles[0][first_mask] + self.instance.landing_takeoff_cost)
            trial += second_multiplier * (profiles[1][second_mask] + self.instance.landing_takeoff_cost)
            if trial + EPSILON < best_cost:
                best_cost, best_mask = trial, first_mask
        stages = list(candidate.edge_stages)
        for local_edge, edge_index in enumerate(mutable):
            stages[edge_index] = first_stage if best_mask & (1 << local_edge) else second_stage
        repaired = Candidate(candidate.delivery_order, tuple(stages))
        # The subset profiles are exact only under their frozen skeletons.  The
        # normal decoder is permitted to improve every stage order, so retain a
        # proposal only when its actual decoded objective improves too.  This
        # keeps this intensification operator from spending ALNS's annealing
        # budget on a skeleton-model artefact.
        if repaired == candidate:
            return candidate
        return repaired if self.evaluate(repaired).objective + EPSILON < evaluation.objective else candidate

    def _stage_ruin_repair(self, candidate: Candidate) -> Candidate:
        """Exactly reassign a bounded set of nearby edges across two stages.

        This is exact only for the selected edge-stage neighbourhood.  Supplying
        a branch-and-cut-backed ``repair_oracle`` to the constructor enlarges
        that subproblem without changing ALNS itself.
        """
        if not candidate.edge_stages or self.instance.stage_count < 2:
            return candidate
        first = self.rng.randrange(self.instance.stage_count - 1)
        allowed_stages = (first, first + 1)
        affected = [
            index for index, stage in enumerate(candidate.edge_stages) if stage in allowed_stages
        ]
        if not affected:
            return candidate
        seed_index = self.rng.choice(affected)
        seed_edge = self.instance.required_edges[seed_index]

        def relatedness(index: int) -> float:
            edge = self.instance.required_edges[index]
            return min(
                self.instance.cost(seed_edge.u, edge.u),
                self.instance.cost(seed_edge.u, edge.v),
                self.instance.cost(seed_edge.v, edge.u),
                self.instance.cost(seed_edge.v, edge.v),
            )

        mutable = tuple(sorted(affected, key=relatedness)[: self.config.local_repair_max_edges])
        global_deadline = self._search_deadline or float("inf")
        deadline = min(global_deadline, monotonic() + self.config.local_repair_seconds)
        return self.repair_oracle.repair(
            candidate,
            self.evaluate,
            mutable_edge_indices=mutable,
            allowed_stages=allowed_stages,
            deadline=deadline,
        )

    def _deadline_reached(self) -> bool:
        return self._search_deadline is not None and monotonic() >= self._search_deadline
