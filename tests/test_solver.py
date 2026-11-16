from __future__ import annotations

from random import Random
from time import monotonic

import pytest

from lddgrp import (
    ALNSConfig,
    ALNSSolver,
    Candidate,
    EnumerativeEdgeStageRepair,
    Evaluation,
    GiantTour,
    LDdGRPInstance,
    RequiredEdge,
)


def metric_line(points: list[float]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(abs(left - right) for right in points) for left in points)


def test_single_edge_exact_stage_decoder_returns_a_valid_minimum_cost_certificate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2]),
        delivery_demands={},
        required_edges=(RequiredEdge(1, 2, 1.0),),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    result = solver.evaluate(Candidate((), (0,)))

    assert result.objective == 4.0
    assert [step.kind for step in result.stages[0].steps] == ["deadhead", "service", "deadhead"]
    solver.validate_evaluation(result)


def test_load_multiplier_is_reduced_after_a_delivery() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2]),
        delivery_demands={1: 5.0},
        required_edges=(RequiredEdge(1, 2, 1.0),),
        curb_weight=2.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    # Service after delivery: stage 0 is 0 -> 1, stage 1 is 1 -> 0 through edge 1--2.
    result = solver.evaluate(Candidate((1,), (1,)))

    assert [stage.multiplier for stage in result.stages] == [7.0, 2.0]
    assert result.stages[1].edge_indices == (0,)
    solver.validate_evaluation(result)


def test_alns_returns_a_complete_feasible_solution() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2, 4, 6]),
        delivery_demands={1: 7.0, 4: 2.0},
        required_edges=(
            RequiredEdge(0, 2, 2.0),
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(1, 2, 1.0),
            RequiredEdge(3, 4, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.2,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=150, seed=11))
    result = solver.solve()

    assert set(result.candidate.delivery_order) == {1, 4}
    assert len(result.candidate.edge_stages) == 4
    assert result.objective > 0
    solver.validate_evaluation(result)


def test_sequence_split_final_repair_receives_its_reserved_time_window() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2, 4]),
        delivery_demands={1: 3.0},
        required_edges=(RequiredEdge(2, 3, 2.0),),
        curb_weight=2.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(
            time_limit_seconds=0.1,
            max_iterations=1,
            seed=3,
            enable_sequence_split_vnd=True,
            sequence_split_seconds=0.001,
            sequence_split_final_seconds=0.04,
        ),
    )
    observed: dict[str, float] = {}

    def inspect_final_window(candidate: Candidate) -> Candidate:
        observed["called_at"] = monotonic()
        assert solver._search_deadline is not None
        observed["deadline"] = solver._search_deadline
        return candidate

    solver._sequence_split_vnd = inspect_final_window  # type: ignore[method-assign]
    result = solver.solve()

    assert observed["deadline"] > observed["called_at"]
    solver.validate_evaluation(result)


def test_giant_tour_exact_dp_projects_a_valid_candidate_and_certificate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 6, 10]),
        delivery_demands={1: 5.0, 4: 2.0},
        required_edges=(RequiredEdge(2, 3, 3.0), RequiredEdge(1, 2, 2.0)),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    # Edge tasks are 0, 1; delivery tasks are 2 (node 1), 3 (node 4).
    fixed = solver.evaluate_giant_tour(GiantTour((0, 2, 1, 3)), dominate_with_decoder=False)
    dominated = solver.evaluate_giant_tour(GiantTour((0, 2, 1, 3)))

    assert fixed.candidate == Candidate((1, 4), (0, 1))
    assert dominated.objective <= fixed.objective
    solver.validate_evaluation(fixed)
    solver.validate_evaluation(dominated)


def test_giant_tour_dp_handles_no_delivery_or_no_required_edge() -> None:
    edge_only = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3]),
        delivery_demands={},
        required_edges=(RequiredEdge(1, 2, 2.0),),
        curb_weight=1.0,
        landing_takeoff_cost=0.5,
    )
    edge_solver = ALNSSolver(edge_only, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    edge_tour = edge_solver.evaluate_giant_tour(GiantTour((0,)), dominate_with_decoder=False)
    assert edge_tour.objective == pytest.approx(edge_solver.evaluate(Candidate((), (0,))).objective)

    delivery_only = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 2]),
        delivery_demands={1: 3.0},
        required_edges=(),
        curb_weight=1.0,
        landing_takeoff_cost=0.5,
    )
    delivery_solver = ALNSSolver(delivery_only, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    delivery_tour = delivery_solver.evaluate_giant_tour(GiantTour((0,)), dominate_with_decoder=False)
    assert delivery_tour.objective == pytest.approx(delivery_solver.evaluate(Candidate((1,), ())).objective)


def test_giant_tour_round_trip_preserves_an_existing_route_certificate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2, 4, 7, 11, 16]),
        delivery_demands={1: 7.0, 4: 2.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(5, 6, 5.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, exact_stage_limit=2, seed=3),
    )
    original = solver.evaluate(Candidate((1, 4), (0, 0, 1, 1, 2)))
    tour = solver.giant_tour_from_evaluation(original)
    round_tripped = solver.evaluate_giant_tour(tour, dominate_with_decoder=False)

    assert round_tripped.objective <= original.objective + 1e-9
    solver.validate_evaluation(round_tripped)


def test_giant_tour_fixed_dp_handles_random_orders_and_empty_stages() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12]),
        delivery_demands={1: 6.0, 4: 2.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1, seed=9))
    task_ids = list(range(len(instance.required_edges) + len(instance.delivery_nodes)))
    rng = Random(17)

    for _ in range(50):
        rng.shuffle(task_ids)
        fixed = solver.evaluate_giant_tour(GiantTour(tuple(task_ids)), dominate_with_decoder=False)
        dominated = solver.evaluate_giant_tour(GiantTour(tuple(task_ids)))
        assert dominated.objective <= fixed.objective + 1e-9
        solver.validate_evaluation(fixed)
        solver.validate_evaluation(dominated)


def test_giant_tour_incremental_relocate_scan_matches_brute_force() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 6, 10]),
        delivery_demands={1: 5.0, 4: 2.0},
        required_edges=(
            RequiredEdge(2, 3, 3.0),
            RequiredEdge(1, 2, 2.0),
            RequiredEdge(3, 4, 4.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1, seed=9))
    task_ids = list(range(len(instance.required_edges) + len(instance.delivery_nodes)))
    rng = Random(31)

    for _ in range(25):
        rng.shuffle(task_ids)
        tour = GiantTour(tuple(task_ids))
        brute_score, _ = solver._giant_tour_orientation_dp(tour)
        tasks = tour.tasks
        for first in range(len(tasks)):
            for size in range(1, min(3, len(tasks) - first) + 1):
                last = first + size - 1
                block = tasks[first : last + 1]
                for insertion_after in range(last + 1, len(tasks)):
                    moved = GiantTour(
                        tasks[:first]
                        + tasks[last + 1 : insertion_after + 1]
                        + block
                        + tasks[insertion_after + 1 :]
                    )
                    brute_score = min(brute_score, solver._giant_tour_orientation_dp(moved)[0])
                for insertion_before in range(first):
                    moved = GiantTour(
                        tasks[:insertion_before]
                        + block
                        + tasks[insertion_before:first]
                        + tasks[last + 1 :]
                    )
                    brute_score = min(brute_score, solver._giant_tour_orientation_dp(moved)[0])
        best_tour, incremental_score = solver._best_giant_tour_relocate(tour, max_block_size=3)

        assert incremental_score == pytest.approx(brute_score, abs=1e-9)
        assert solver._giant_tour_orientation_dp(best_tour)[0] == pytest.approx(brute_score, abs=1e-9)


def test_giant_tour_vnd_never_worsens_a_certified_evaluation() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12, 16]),
        delivery_demands={1: 7.0, 4: 2.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(5, 6, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, giant_tour_max_passes=2, giant_tour_max_block_size=2),
    )
    original = solver.evaluate(Candidate((1, 4), (0, 1, 2, 1, 0)))
    improved = solver._giant_tour_vnd(original)

    assert improved.objective <= original.objective
    solver.validate_evaluation(improved)


def test_giant_tour_time_cap_scales_down_for_small_instances() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2]),
        delivery_demands={1: 1.0},
        required_edges=(RequiredEdge(1, 2, 1.0),),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, giant_tour_final_seconds=0.8),
    )

    assert solver._giant_tour_time_cap() == pytest.approx(0.01)


def test_certificate_rejects_a_service_in_the_wrong_candidate_stage() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2]),
        delivery_demands={1: 1.0},
        required_edges=(RequiredEdge(1, 2, 1.0),),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    valid = solver.evaluate(Candidate((1,), (0,)))
    malformed = Evaluation(Candidate((1,), (1,)), valid.objective, valid.stages)

    with pytest.raises(ValueError, match="different candidate stage"):
        solver.validate_evaluation(malformed)


def test_enumerative_repair_finds_the_best_assignment_in_its_neighbourhood() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 10]),
        delivery_demands={1: 9.0},
        required_edges=(RequiredEdge(2, 3, 7.0),),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    incumbent = Candidate((1,), (0,))
    repaired = EnumerativeEdgeStageRepair(max_mutable_edges=1).repair(
        incumbent,
        solver.evaluate,
        mutable_edge_indices=(0,),
        allowed_stages=(0, 1),
        deadline=monotonic() + 1.0,
    )

    assert repaired.edge_stages == (1,)
    assert solver.evaluate(repaired).objective < solver.evaluate(incumbent).objective


def test_delivery_ruin_recreate_only_evaluates_complete_candidates() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 6]),
        delivery_demands={1: 1.0, 2: 2.0, 3: 3.0},
        required_edges=(RequiredEdge(1, 2, 2.0),),
        curb_weight=2.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1, seed=4))
    original = Candidate((1, 2, 3), (0,))

    repaired = solver._delivery_ruin_recreate(original)

    solver.validate_candidate(repaired)
    solver.validate_evaluation(solver.evaluate(repaired))


def test_best_single_edge_reassignment_does_not_worsen_the_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 10]),
        delivery_demands={1: 9.0},
        required_edges=(RequiredEdge(2, 3, 7.0),),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1, seed=2))
    incumbent = Candidate((1,), (0,))

    improved = solver._edge_best_reassign(incumbent)

    assert solver.evaluate(improved).objective <= solver.evaluate(incumbent).objective
    solver.validate_evaluation(solver.evaluate(improved))


def test_coupled_initializer_returns_a_valid_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2, 4, 7, 11]),
        delivery_demands={1: 4.0, 4: 2.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 5, 7.0),
            RequiredEdge(1, 2, 1.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1, seed=3))

    candidate = solver.initial_candidate()

    solver.validate_candidate(candidate)
    solver.validate_evaluation(solver.evaluate(candidate))


def test_edge_cluster_recreate_returns_a_complete_feasible_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12]),
        delivery_demands={1: 4.0, 4: 2.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1, seed=9))
    original = Candidate((1, 4), (0, 1, 2, 0))

    recreated = solver._edge_cluster_recreate(original)

    solver.validate_candidate(recreated)
    solver.validate_evaluation(solver.evaluate(recreated))


def test_large_stage_decoder_improves_a_crossing_service_order() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        delivery_demands={8: 1.0},
        required_edges=(
            RequiredEdge(1, 2, 1.0),
            RequiredEdge(7, 8, 1.0),
            RequiredEdge(3, 4, 1.0),
        ),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, exact_stage_limit=2, large_stage_local_search_passes=3),
    )
    initial = solver._best_orientations_for_order(0, 8, (0, 1, 2), (0, 1, 2))
    improved = solver._improve_stage_order(0, 8, (0, 1, 2), (0, 1, 2))

    assert sum(step.cost for step in improved) < sum(step.cost for step in initial)
    evaluated = solver.evaluate(Candidate((8,), (0, 0, 0)))
    solver.validate_evaluation(evaluated)


def test_edge_cluster_stage_descent_never_worsens_a_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12]),
        delivery_demands={1: 6.0, 4: 1.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, stage_descent_cluster_size=4, seed=9),
    )
    original = Candidate((1, 4), (0, 0, 0, 0))

    descended = solver._edge_cluster_stage_descent(original)

    assert solver.evaluate(descended).objective <= solver.evaluate(original).objective
    solver.validate_evaluation(solver.evaluate(descended))


def test_load_aware_ruin_recreate_never_worsens_a_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12]),
        delivery_demands={1: 6.0, 4: 1.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, load_aware_ruin_max_edges=3, seed=5),
    )
    original = Candidate((1, 4), (0, 0, 0, 0))

    repaired = solver._load_aware_ruin_recreate(original)

    assert solver.evaluate(repaired).objective <= solver.evaluate(original).objective
    solver.validate_evaluation(solver.evaluate(repaired))


def test_route_block_relocate_never_worsens_a_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12]),
        delivery_demands={1: 6.0, 4: 1.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, route_block_max_edges=3, seed=5),
    )
    original = Candidate((1, 4), (0, 0, 0, 0))

    relocated = solver._route_block_relocate(original)

    assert solver.evaluate(relocated).objective <= solver.evaluate(original).objective
    solver.validate_evaluation(solver.evaluate(relocated))


def test_sequence_split_vnd_returns_a_valid_complete_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12, 16]),
        delivery_demands={1: 7.0, 4: 2.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(5, 6, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, sequence_split_max_moves=4, seed=5),
    )
    original = Candidate((1, 4), (0, 1, 2, 1, 0))

    split = solver._sequence_split_vnd(original)

    solver.validate_candidate(split)
    solver.validate_evaluation(solver.evaluate(split))


def test_spatial_two_stage_exact_repair_returns_a_complete_feasible_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12]),
        delivery_demands={1: 6.0, 4: 1.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, spatial_repair_max_edges=4, seed=2),
    )
    original = Candidate((1, 4), (0, 1, 2, 0))

    repaired = solver._spatial_two_stage_exact_repair(original)

    solver.validate_candidate(repaired)
    solver.validate_evaluation(solver.evaluate(repaired))


def test_frozen_skeleton_profile_matches_the_exact_stage_decoder_without_fixed_services() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8]),
        delivery_demands={4: 1.0},
        required_edges=(
            RequiredEdge(1, 2, 2.0),
            RequiredEdge(2, 3, 2.0),
        ),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1))

    profile = solver._frozen_skeleton_stage_subset_costs(
        0,
        4,
        (),
        (0, 1),
        monotonic() + 1.0,
    )

    assert profile is not None
    decoded = solver._decode_stage(0, 4, (0, 1))
    assert profile[3] == sum(step.cost for step in decoded)


def test_skeleton_two_stage_exact_repair_returns_a_complete_feasible_candidate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 3, 5, 8, 12]),
        delivery_demands={1: 6.0, 4: 1.0},
        required_edges=(
            RequiredEdge(2, 3, 2.0),
            RequiredEdge(3, 4, 3.0),
            RequiredEdge(4, 5, 4.0),
            RequiredEdge(1, 2, 2.0),
        ),
        curb_weight=3.0,
        landing_takeoff_cost=0.5,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(time_limit_seconds=1, max_iterations=1, skeleton_repair_max_edges=4, seed=2),
    )
    original = Candidate((1, 4), (0, 1, 2, 0))

    repaired = solver._skeleton_two_stage_exact_repair(original)

    assert solver.evaluate(repaired).objective <= solver.evaluate(original).objective
    solver.validate_candidate(repaired)
    solver.validate_evaluation(solver.evaluate(repaired))


def test_two_tier_search_returns_a_high_fidelity_valid_certificate() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        delivery_demands={8: 1.0},
        required_edges=(
            RequiredEdge(1, 2, 1.0),
            RequiredEdge(7, 8, 1.0),
            RequiredEdge(3, 4, 1.0),
        ),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(
        instance,
        ALNSConfig(
            time_limit_seconds=1,
            max_iterations=10,
            exact_stage_limit=2,
            large_stage_local_search_passes=3,
            search_large_stage_local_search_passes=0,
        ),
    )

    result = solver.solve()

    assert solver._active_large_stage_local_search_passes == 3
    solver.validate_evaluation(result)


def test_delivery_order_dp_finds_the_best_order_for_fixed_stage_labels() -> None:
    instance = LDdGRPInstance(
        depot=0,
        travel_cost=metric_line([0, 1, 4, 5, 8]),
        delivery_demands={1: 1.0, 4: 9.0},
        required_edges=(RequiredEdge(2, 3, 1.0),),
        curb_weight=1.0,
        landing_takeoff_cost=0.0,
    )
    solver = ALNSSolver(instance, ALNSConfig(time_limit_seconds=1, max_iterations=1))
    original = Candidate((1, 4), (0,))

    optimized = solver._delivery_order_dp(original)

    assert optimized.delivery_order == (4, 1)
    assert solver.evaluate(optimized).objective < solver.evaluate(original).objective
    solver.validate_evaluation(solver.evaluate(optimized))

