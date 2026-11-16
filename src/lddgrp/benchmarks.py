"""Readers and a runner for the authors' published LDdGRP benchmark set.

The data archive and results file use the generic routing-problem text format.
In the LDdGRP experiments the delivery-node ``coste`` field is not part of the
Chapter 7 objective: the published results file supplies the instance-specific
``w_0`` and ``c_D`` parameters.  The parser retains that field as metadata so
the distinction is explicit rather than silently assuming it is ``c_D``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from math import hypot
from pathlib import Path
from time import monotonic
from typing import Iterable, Mapping

from .cli import evaluation_to_dict
from .model import LDdGRPInstance, RequiredEdge
from .solver import ALNSConfig, ALNSSolver


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_EDGE = re.compile(rf"^\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+coste\s+({_NUMBER})\s*$")
_DELIVERY = re.compile(rf"^(\d+)\s+coste\s+({_NUMBER})\s+demanda\s+({_NUMBER})\s*$")


@dataclass(frozen=True, slots=True)
class PublishedResult:
    """Parameters and reference bounds published with one official instance."""

    instance_name: str
    curb_weight: float
    landing_takeoff_cost: float
    lower_bound: float | None
    upper_bound: float
    is_optimal: bool


@dataclass(frozen=True, slots=True)
class OfficialBenchmark:
    """An official instance plus parameters supplied in the results file."""

    source_name: str
    instance: LDdGRPInstance
    published: PublishedResult
    node_labels_by_internal_index: tuple[int, ...]
    generic_delivery_costs: Mapping[int, float]


def read_published_results(path: Path) -> dict[str, PublishedResult]:
    """Read the authors' ``LDdGRP results.txt`` table.

    Its rows contain either an upper bound alone (proved optimal) or a lower
    and upper bound (not proved optimal).  A malformed data row raises instead
    of guessing which number is which.
    """

    records: dict[str, PublishedResult] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = raw_line.split()
        if not parts or parts[0] == "Instance" or set(raw_line.strip()) == {"-"}:
            continue
        if len(parts) < 5 or not parts[0].endswith(".dat"):
            raise ValueError(f"Cannot parse published-result row: {raw_line!r}")
        if parts[-1] == "optimo":
            bound_tokens = parts[3:-1]
            is_optimal = True
        elif len(parts) >= 6 and parts[-2:] == ["no", "opt"]:
            bound_tokens = parts[3:-2]
            is_optimal = False
        else:
            raise ValueError(f"Unknown published-result status: {raw_line!r}")
        if len(bound_tokens) == 1:
            lower_bound = None
            upper_bound = float(bound_tokens[0])
        elif len(bound_tokens) == 2:
            lower_bound = float(bound_tokens[0])
            upper_bound = float(bound_tokens[1])
        else:
            raise ValueError(f"Expected one or two bounds in row: {raw_line!r}")
        name = parts[0]
        if name in records:
            raise ValueError(f"Duplicate published-result row for {name!r}.")
        records[name] = PublishedResult(
            instance_name=name,
            curb_weight=float(parts[1]),
            landing_takeoff_cost=float(parts[2]),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_optimal=is_optimal,
        )
    if not records:
        raise ValueError("The published results file did not contain any instances.")
    return records


def load_official_benchmark(instance_path: Path, published: Mapping[str, PublishedResult]) -> OfficialBenchmark:
    """Load one official ``.dat`` file and its published LDdGRP parameters."""

    try:
        result = published[instance_path.name]
    except KeyError as error:
        raise ValueError(f"No published parameters found for {instance_path.name!r}.") from error

    lines = [line.strip() for line in instance_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    declared_name = _value_after_colon(lines, "NOMBRE")
    if declared_name != instance_path.name:
        raise ValueError(f"Instance filename and NOMBRE disagree in {instance_path}.")
    depot_label = int(_value_after_colon(lines, "DEPOT"))
    vertex_count = int(_value_after_colon(lines, "VERTICES"))
    edge_count = int(_value_after_colon(lines, "ARISTAS_REQ"))
    delivery_count = int(_value_after_colon(lines, "VERTICES_REQ"))

    try:
        edge_header = lines.index("LISTA_ARISTAS_REQ :")
        delivery_header = lines.index("LISTA_VERTICES_REQ :")
        coordinate_header = lines.index("COORDENADAS :")
    except ValueError as error:
        raise ValueError(f"Missing a section header in {instance_path}.") from error
    if not edge_header < delivery_header < coordinate_header:
        raise ValueError(f"Sections are out of order in {instance_path}.")

    edges: list[tuple[int, int, float]] = []
    for line in lines[edge_header + 1 : delivery_header]:
        match = _EDGE.match(line)
        if match is None:
            raise ValueError(f"Cannot parse required-edge line {line!r} in {instance_path}.")
        edges.append((int(match.group(1)), int(match.group(2)), float(match.group(3))))
    if len(edges) != edge_count:
        raise ValueError(f"Expected {edge_count} required edges in {instance_path}, found {len(edges)}.")

    generic_delivery_costs: dict[int, float] = {}
    demands_by_label: dict[int, float] = {}
    for line in lines[delivery_header + 1 : coordinate_header]:
        match = _DELIVERY.match(line)
        if match is None:
            raise ValueError(f"Cannot parse required-vertex line {line!r} in {instance_path}.")
        label = int(match.group(1))
        if label in demands_by_label:
            raise ValueError(f"Duplicate required vertex {label} in {instance_path}.")
        generic_delivery_costs[label] = float(match.group(2))
        demands_by_label[label] = float(match.group(3))
    if len(demands_by_label) != delivery_count:
        raise ValueError(
            f"Expected {delivery_count} required vertices in {instance_path}, found {len(demands_by_label)}."
        )

    coordinates: dict[int, tuple[float, float]] = {}
    for line in lines[coordinate_header + 1 :]:
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Cannot parse coordinate line {line!r} in {instance_path}.")
        label = int(fields[0])
        if label in coordinates:
            raise ValueError(f"Duplicate coordinate for vertex {label} in {instance_path}.")
        coordinates[label] = (float(fields[1]), float(fields[2]))
    if len(coordinates) != vertex_count:
        raise ValueError(f"Expected {vertex_count} coordinates in {instance_path}, found {len(coordinates)}.")
    if depot_label not in coordinates:
        raise ValueError(f"Depot {depot_label} has no coordinate in {instance_path}.")

    labels = tuple(sorted(coordinates))
    node_index = {label: index for index, label in enumerate(labels)}
    missing = {label for edge in edges for label in edge[:2]} | set(demands_by_label)
    if not missing <= node_index.keys():
        raise ValueError(f"An edge or delivery node has no coordinate in {instance_path}.")
    travel_cost = tuple(
        tuple(hypot(coordinates[left][0] - coordinates[right][0], coordinates[left][1] - coordinates[right][1]) for right in labels)
        for left in labels
    )
    instance = LDdGRPInstance(
        depot=node_index[depot_label],
        travel_cost=travel_cost,
        delivery_demands={node_index[label]: demand for label, demand in demands_by_label.items()},
        required_edges=tuple(RequiredEdge(node_index[left], node_index[right], cost) for left, right, cost in edges),
        curb_weight=result.curb_weight,
        landing_takeoff_cost=result.landing_takeoff_cost,
    )
    return OfficialBenchmark(
        source_name=instance_path.name,
        instance=instance,
        published=result,
        node_labels_by_internal_index=labels,
        generic_delivery_costs=generic_delivery_costs,
    )


def _value_after_colon(lines: Iterable[str], key: str) -> str:
    prefix = f"{key} :"
    matches = [line[len(prefix) :].strip() for line in lines if line.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"Expected exactly one non-empty {key} field.")
    return matches[0]


def _run_record(benchmark: OfficialBenchmark, config: ALNSConfig, *, restarts: int) -> dict[str, object]:
    if restarts < 1:
        raise ValueError("restarts must be at least one.")
    started = monotonic()
    restart_seconds = config.time_limit_seconds / restarts
    evaluations = []
    for restart in range(restarts):
        restart_config = ALNSConfig(
            time_limit_seconds=restart_seconds,
            max_iterations=config.max_iterations,
            exact_stage_limit=config.exact_stage_limit,
            seed=config.seed + restart,
            initial_temperature_fraction=config.initial_temperature_fraction,
            cooling_rate=config.cooling_rate,
            adaptation_period=config.adaptation_period,
            local_repair_max_edges=config.local_repair_max_edges,
            local_repair_seconds=config.local_repair_seconds,
            initialization_fraction=config.initialization_fraction,
            large_stage_local_search_passes=config.large_stage_local_search_passes,
            search_large_stage_local_search_passes=config.search_large_stage_local_search_passes,
            finalization_fraction=config.finalization_fraction,
            stage_descent_cluster_size=config.stage_descent_cluster_size,
            stage_descent_seconds=config.stage_descent_seconds,
            stage_descent_passes=config.stage_descent_passes,
            enable_stage_descent=config.enable_stage_descent,
            spatial_repair_max_edges=config.spatial_repair_max_edges,
            spatial_repair_seconds=config.spatial_repair_seconds,
            enable_spatial_repair=config.enable_spatial_repair,
            skeleton_repair_max_edges=config.skeleton_repair_max_edges,
            skeleton_repair_seconds=config.skeleton_repair_seconds,
            enable_skeleton_repair=config.enable_skeleton_repair,
            enable_delivery_order_dp=config.enable_delivery_order_dp,
            load_aware_ruin_max_edges=config.load_aware_ruin_max_edges,
            load_aware_ruin_seconds=config.load_aware_ruin_seconds,
            load_aware_final_seconds=config.load_aware_final_seconds / restarts,
            load_aware_final_passes=config.load_aware_final_passes,
            enable_load_aware_ruin=config.enable_load_aware_ruin,
            route_block_max_edges=config.route_block_max_edges,
            route_block_seconds=config.route_block_seconds,
            enable_route_block_relocate=config.enable_route_block_relocate,
            sequence_split_seconds=config.sequence_split_seconds,
            sequence_split_max_moves=config.sequence_split_max_moves,
            sequence_split_final_seconds=config.sequence_split_final_seconds / restarts,
            enable_sequence_split_vnd=config.enable_sequence_split_vnd,
            giant_tour_final_seconds=config.giant_tour_final_seconds / restarts,
            giant_tour_max_passes=config.giant_tour_max_passes,
            giant_tour_max_block_size=config.giant_tour_max_block_size,
            enable_giant_tour_vnd=config.enable_giant_tour_vnd,
        )
        evaluations.append(ALNSSolver(benchmark.instance, restart_config).solve())
    evaluation = min(evaluations, key=lambda result: result.objective)
    elapsed = monotonic() - started
    reference = benchmark.published
    return {
        "instance": benchmark.source_name,
        "curb_weight": reference.curb_weight,
        "landing_takeoff_cost": reference.landing_takeoff_cost,
        "published_lower_bound": reference.lower_bound,
        "published_upper_bound": reference.upper_bound,
        "published_optimal": reference.is_optimal,
        "heuristic_objective": evaluation.objective,
        "gap_to_published_upper_bound_percent": 100.0
        * (evaluation.objective - reference.upper_bound)
        / reference.upper_bound,
        "elapsed_seconds": elapsed,
        "restart_count": restarts,
        "restart_objectives": [result.objective for result in evaluations],
        "node_labels_by_internal_index": list(benchmark.node_labels_by_internal_index),
        "generic_delivery_costs_not_used_by_lddgrp_objective": dict(benchmark.generic_delivery_costs),
        "certificate": evaluation_to_dict(evaluation),
    }


def _write_report(
    output: Path,
    *,
    records: list[dict[str, object]],
    seconds_per_instance: float,
    iterations_per_instance: int,
    base_seed: int,
    restarts: int,
    exact_stage_limit: int,
    large_stage_local_search_passes: int,
    skeleton_repair_max_edges: int,
    skeleton_repair_seconds: float,
    enable_skeleton_repair: bool,
    enable_delivery_order_dp: bool,
    load_aware_ruin_max_edges: int,
    load_aware_ruin_seconds: float,
    load_aware_final_seconds: float,
    load_aware_final_passes: int,
    enable_load_aware_ruin: bool,
    route_block_max_edges: int,
    route_block_seconds: float,
    enable_route_block_relocate: bool,
    sequence_split_seconds: float,
    sequence_split_max_moves: int,
    sequence_split_final_seconds: float,
    enable_sequence_split_vnd: bool,
    giant_tour_final_seconds: float,
    giant_tour_max_passes: int,
    giant_tour_max_block_size: int,
    enable_giant_tour_vnd: bool,
) -> None:
    """Write a resumable report after a complete instance result is available."""

    report = {
        "method": "stage-aware ALNS with exact small-stage decoding and bounded enumerative local repair",
        "instance_count": len(records),
        "seconds_per_instance": seconds_per_instance,
        "iterations_per_instance": iterations_per_instance,
        "base_seed": base_seed,
        "restarts": restarts,
        "exact_stage_limit": exact_stage_limit,
        "large_stage_local_search_passes": large_stage_local_search_passes,
        "skeleton_repair_max_edges": skeleton_repair_max_edges,
        "skeleton_repair_seconds": skeleton_repair_seconds,
        "enable_skeleton_repair": enable_skeleton_repair,
        "enable_delivery_order_dp": enable_delivery_order_dp,
        "load_aware_ruin_max_edges": load_aware_ruin_max_edges,
        "load_aware_ruin_seconds": load_aware_ruin_seconds,
        "load_aware_final_seconds": load_aware_final_seconds,
        "load_aware_final_passes": load_aware_final_passes,
        "enable_load_aware_ruin": enable_load_aware_ruin,
        "route_block_max_edges": route_block_max_edges,
        "route_block_seconds": route_block_seconds,
        "enable_route_block_relocate": enable_route_block_relocate,
        "sequence_split_seconds": sequence_split_seconds,
        "sequence_split_max_moves": sequence_split_max_moves,
        "sequence_split_final_seconds": sequence_split_final_seconds,
        "enable_sequence_split_vnd": enable_sequence_split_vnd,
        "giant_tour_final_seconds": giant_tour_final_seconds,
        "giant_tour_max_passes": giant_tour_max_passes,
        "giant_tour_max_block_size": giant_tour_max_block_size,
        "enable_giant_tour_vnd": enable_giant_tour_vnd,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALNS on the authors' official LDdGRP benchmark files.")
    parser.add_argument("instance_directory", type=Path, help="Directory containing official .dat files.")
    parser.add_argument("results", type=Path, help="Authors' LDdGRP results.txt parameter table.")
    parser.add_argument("--pattern", default="*.dat", help="Filename glob, default: *.dat.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Per-instance ALNS wall-clock limit.")
    parser.add_argument("--iterations", type=int, default=5_000, help="Per-instance iteration limit.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--exact-stage-limit", type=int, default=10, help="Exact per-stage DP edge limit.")
    parser.add_argument(
        "--large-stage-local-search-passes",
        type=int,
        default=12,
        help="2-opt/relocation passes after large-stage construction (default: 12).",
    )
    parser.add_argument(
        "--enable-delivery-order-dp",
        action="store_true",
        help="Run the exact fixed-edge-stage delivery-order subproblem as an ALNS operator (ablation).",
    )
    parser.add_argument(
        "--enable-skeleton-repair",
        action="store_true",
        help="Run the exact frozen-route-skeleton two-stage repair as an ALNS operator (ablation).",
    )
    parser.add_argument(
        "--skeleton-repair-max-edges",
        type=int,
        default=7,
        help="Maximum mutable-edge count for frozen-skeleton repair (default: 7).",
    )
    parser.add_argument(
        "--skeleton-repair-seconds",
        type=float,
        default=0.08,
        help="Per-invocation frozen-skeleton repair cap in seconds (default: 0.08).",
    )
    parser.add_argument(
        "--enable-load-aware-ruin",
        action="store_true",
        help="Use load-weighted bad-edge removal and regret reinsertion (ablation).",
    )
    parser.add_argument(
        "--load-aware-ruin-max-edges",
        type=int,
        default=6,
        help="Maximum edges destroyed by load-aware ruin (default: 6).",
    )
    parser.add_argument(
        "--load-aware-ruin-seconds",
        type=float,
        default=0.06,
        help="Per-invocation load-aware ruin cap in seconds (default: 0.06).",
    )
    parser.add_argument(
        "--load-aware-final-seconds",
        type=float,
        default=0.08,
        help="Reserved final load-aware repair window in seconds (default: 0.08).",
    )
    parser.add_argument(
        "--load-aware-final-passes",
        type=int,
        default=1,
        help="Final load-aware repair passes when enabled (default: 1).",
    )
    parser.add_argument(
        "--enable-route-block-relocate",
        action="store_true",
        help="Use a decoded-route block relocation neighbourhood (ablation).",
    )
    parser.add_argument(
        "--route-block-max-edges",
        type=int,
        default=4,
        help="Maximum services in a relocated route block (default: 4).",
    )
    parser.add_argument(
        "--route-block-seconds",
        type=float,
        default=0.04,
        help="Per-invocation route-block relocation cap in seconds (default: 0.04).",
    )
    parser.add_argument(
        "--enable-sequence-split-vnd",
        action="store_true",
        help="Use global service-sequence split/VNS repair (ablation).",
    )
    parser.add_argument(
        "--sequence-split-seconds",
        type=float,
        default=0.10,
        help="Per-invocation sequence-split VNS cap in seconds (default: 0.10).",
    )
    parser.add_argument(
        "--sequence-split-max-moves",
        type=int,
        default=12,
        help="Maximum sequence perturbations per split-VNS call (default: 12).",
    )
    parser.add_argument(
        "--sequence-split-final-seconds",
        type=float,
        default=0.12,
        help="Reserved final sequence-split repair window in seconds (default: 0.12).",
    )
    parser.add_argument(
        "--enable-giant-tour-vnd",
        action="store_true",
        help="Run the certified giant-tour relocate/Or-opt VND as final intensification (ablation).",
    )
    parser.add_argument(
        "--giant-tour-final-seconds",
        type=float,
        default=0.80,
        help="Reserved final giant-tour VND window in seconds (default: 0.80).",
    )
    parser.add_argument(
        "--giant-tour-max-passes",
        type=int,
        default=2,
        help="Maximum certified giant-tour VND passes (default: 2).",
    )
    parser.add_argument(
        "--giant-tour-max-block-size",
        type=int,
        default=2,
        help="Maximum giant-tour relocate/Or-opt block size (default: 2).",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="Independent ALNS starts sharing each instance's total time budget (default: 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already recorded instances from --output and solve only missing matches.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output benchmark report JSON.")
    args = parser.parse_args()

    if not args.instance_directory.is_dir():
        raise ValueError(f"Not an instance directory: {args.instance_directory}")
    published = read_published_results(args.results)
    paths = sorted(path for path in args.instance_directory.iterdir() if path.is_file() and fnmatch(path.name, args.pattern))
    if not paths:
        raise ValueError(f"No official instances matched {args.pattern!r}.")
    if args.restarts < 1:
        raise ValueError("--restarts must be at least one.")
    if args.large_stage_local_search_passes < 0:
        raise ValueError("--large-stage-local-search-passes must be non-negative.")
    if args.skeleton_repair_max_edges < 1:
        raise ValueError("--skeleton-repair-max-edges must be at least one.")
    if args.skeleton_repair_seconds <= 0:
        raise ValueError("--skeleton-repair-seconds must be positive.")
    if args.load_aware_ruin_max_edges < 1:
        raise ValueError("--load-aware-ruin-max-edges must be at least one.")
    if args.load_aware_ruin_seconds <= 0:
        raise ValueError("--load-aware-ruin-seconds must be positive.")
    if args.load_aware_final_seconds <= 0:
        raise ValueError("--load-aware-final-seconds must be positive.")
    if args.load_aware_final_passes < 1:
        raise ValueError("--load-aware-final-passes must be at least one.")
    if args.route_block_max_edges < 1:
        raise ValueError("--route-block-max-edges must be at least one.")
    if args.route_block_seconds <= 0:
        raise ValueError("--route-block-seconds must be positive.")
    if args.sequence_split_seconds <= 0:
        raise ValueError("--sequence-split-seconds must be positive.")
    if args.sequence_split_max_moves < 0:
        raise ValueError("--sequence-split-max-moves must be non-negative.")
    if args.sequence_split_final_seconds <= 0:
        raise ValueError("--sequence-split-final-seconds must be positive.")
    if args.giant_tour_final_seconds <= 0:
        raise ValueError("--giant-tour-final-seconds must be positive.")
    if args.giant_tour_max_passes < 1:
        raise ValueError("--giant-tour-max-passes must be at least one.")
    if args.giant_tour_max_block_size < 1:
        raise ValueError("--giant-tour-max-block-size must be at least one.")
    reserved_final_seconds = (
        (args.load_aware_final_seconds if args.enable_load_aware_ruin else 0.0)
        + (args.sequence_split_final_seconds if args.enable_sequence_split_vnd else 0.0)
        + (args.giant_tour_final_seconds if args.enable_giant_tour_vnd else 0.0)
    )
    if reserved_final_seconds >= args.seconds:
        raise ValueError("Enabled final-repair reservations must be shorter than --seconds.")
    records_by_name: dict[str, dict[str, object]] = {}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            previous.get("seconds_per_instance") != args.seconds
            or previous.get("iterations_per_instance") != args.iterations
            or previous.get("base_seed") != args.seed
            or previous.get("restarts", 1) != args.restarts
            or previous.get("exact_stage_limit", 10) != args.exact_stage_limit
            or previous.get("large_stage_local_search_passes", 3) != args.large_stage_local_search_passes
            or previous.get("skeleton_repair_max_edges", 7) != args.skeleton_repair_max_edges
            or previous.get("skeleton_repair_seconds", 0.08) != args.skeleton_repair_seconds
            or previous.get("enable_skeleton_repair", False) != args.enable_skeleton_repair
            or previous.get("enable_delivery_order_dp", False) != args.enable_delivery_order_dp
            or previous.get("load_aware_ruin_max_edges", 6) != args.load_aware_ruin_max_edges
            or previous.get("load_aware_ruin_seconds", 0.06) != args.load_aware_ruin_seconds
            or previous.get("load_aware_final_seconds", 0.08) != args.load_aware_final_seconds
            or previous.get("load_aware_final_passes", 1) != args.load_aware_final_passes
            or previous.get("enable_load_aware_ruin", False) != args.enable_load_aware_ruin
            or previous.get("route_block_max_edges", 4) != args.route_block_max_edges
            or previous.get("route_block_seconds", 0.04) != args.route_block_seconds
            or previous.get("enable_route_block_relocate", False) != args.enable_route_block_relocate
            or previous.get("sequence_split_seconds", 0.10) != args.sequence_split_seconds
            or previous.get("sequence_split_max_moves", 12) != args.sequence_split_max_moves
            or previous.get("sequence_split_final_seconds", 0.12) != args.sequence_split_final_seconds
            or previous.get("enable_sequence_split_vnd", False) != args.enable_sequence_split_vnd
            or previous.get("giant_tour_final_seconds", 0.80) != args.giant_tour_final_seconds
            or previous.get("giant_tour_max_passes", 2) != args.giant_tour_max_passes
            or previous.get("giant_tour_max_block_size", 2) != args.giant_tour_max_block_size
            or previous.get("enable_giant_tour_vnd", False) != args.enable_giant_tour_vnd
        ):
            raise ValueError("Cannot resume a report created with different benchmark settings.")
        for record in previous.get("records", []):
            name = record.get("instance")
            if not isinstance(name, str):
                raise ValueError("Cannot resume: an existing record lacks an instance name.")
            records_by_name[name] = record
    for run_index, path in enumerate(paths):
        if path.name in records_by_name:
            print(f"{path.name}: reused")
            continue
        benchmark = load_official_benchmark(path, published)
        config = ALNSConfig(
            time_limit_seconds=args.seconds,
            max_iterations=args.iterations,
            exact_stage_limit=args.exact_stage_limit,
            seed=args.seed + run_index,
            large_stage_local_search_passes=args.large_stage_local_search_passes,
            skeleton_repair_max_edges=args.skeleton_repair_max_edges,
            skeleton_repair_seconds=args.skeleton_repair_seconds,
            enable_skeleton_repair=args.enable_skeleton_repair,
            enable_delivery_order_dp=args.enable_delivery_order_dp,
            load_aware_ruin_max_edges=args.load_aware_ruin_max_edges,
            load_aware_ruin_seconds=args.load_aware_ruin_seconds,
            load_aware_final_seconds=args.load_aware_final_seconds,
            load_aware_final_passes=args.load_aware_final_passes,
            enable_load_aware_ruin=args.enable_load_aware_ruin,
            route_block_max_edges=args.route_block_max_edges,
            route_block_seconds=args.route_block_seconds,
            enable_route_block_relocate=args.enable_route_block_relocate,
            sequence_split_seconds=args.sequence_split_seconds,
            sequence_split_max_moves=args.sequence_split_max_moves,
            sequence_split_final_seconds=args.sequence_split_final_seconds,
            enable_sequence_split_vnd=args.enable_sequence_split_vnd,
            giant_tour_final_seconds=args.giant_tour_final_seconds,
            giant_tour_max_passes=args.giant_tour_max_passes,
            giant_tour_max_block_size=args.giant_tour_max_block_size,
            enable_giant_tour_vnd=args.enable_giant_tour_vnd,
        )
        record = _run_record(benchmark, config, restarts=args.restarts)
        records_by_name[path.name] = record
        records = [records_by_name[path.name] for path in paths if path.name in records_by_name]
        _write_report(
            args.output,
            records=records,
            seconds_per_instance=args.seconds,
            iterations_per_instance=args.iterations,
            base_seed=args.seed,
            restarts=args.restarts,
            exact_stage_limit=args.exact_stage_limit,
            large_stage_local_search_passes=args.large_stage_local_search_passes,
            skeleton_repair_max_edges=args.skeleton_repair_max_edges,
            skeleton_repair_seconds=args.skeleton_repair_seconds,
            enable_skeleton_repair=args.enable_skeleton_repair,
            enable_delivery_order_dp=args.enable_delivery_order_dp,
            load_aware_ruin_max_edges=args.load_aware_ruin_max_edges,
            load_aware_ruin_seconds=args.load_aware_ruin_seconds,
            load_aware_final_seconds=args.load_aware_final_seconds,
            load_aware_final_passes=args.load_aware_final_passes,
            enable_load_aware_ruin=args.enable_load_aware_ruin,
            route_block_max_edges=args.route_block_max_edges,
            route_block_seconds=args.route_block_seconds,
            enable_route_block_relocate=args.enable_route_block_relocate,
            sequence_split_seconds=args.sequence_split_seconds,
            sequence_split_max_moves=args.sequence_split_max_moves,
            sequence_split_final_seconds=args.sequence_split_final_seconds,
            enable_sequence_split_vnd=args.enable_sequence_split_vnd,
            giant_tour_final_seconds=args.giant_tour_final_seconds,
            giant_tour_max_passes=args.giant_tour_max_passes,
            giant_tour_max_block_size=args.giant_tour_max_block_size,
            enable_giant_tour_vnd=args.enable_giant_tour_vnd,
        )
        print(f"{path.name}: objective={record['heuristic_objective']:.6f}")
    records = [records_by_name[path.name] for path in paths if path.name in records_by_name]
    _write_report(
        args.output,
        records=records,
        seconds_per_instance=args.seconds,
        iterations_per_instance=args.iterations,
        base_seed=args.seed,
        restarts=args.restarts,
        exact_stage_limit=args.exact_stage_limit,
        large_stage_local_search_passes=args.large_stage_local_search_passes,
        skeleton_repair_max_edges=args.skeleton_repair_max_edges,
        skeleton_repair_seconds=args.skeleton_repair_seconds,
        enable_skeleton_repair=args.enable_skeleton_repair,
        enable_delivery_order_dp=args.enable_delivery_order_dp,
        load_aware_ruin_max_edges=args.load_aware_ruin_max_edges,
        load_aware_ruin_seconds=args.load_aware_ruin_seconds,
        load_aware_final_seconds=args.load_aware_final_seconds,
        load_aware_final_passes=args.load_aware_final_passes,
        enable_load_aware_ruin=args.enable_load_aware_ruin,
        route_block_max_edges=args.route_block_max_edges,
        route_block_seconds=args.route_block_seconds,
        enable_route_block_relocate=args.enable_route_block_relocate,
        sequence_split_seconds=args.sequence_split_seconds,
        sequence_split_max_moves=args.sequence_split_max_moves,
        sequence_split_final_seconds=args.sequence_split_final_seconds,
        enable_sequence_split_vnd=args.enable_sequence_split_vnd,
        giant_tour_final_seconds=args.giant_tour_final_seconds,
        giant_tour_max_passes=args.giant_tour_max_passes,
        giant_tour_max_block_size=args.giant_tour_max_block_size,
        enable_giant_tour_vnd=args.enable_giant_tour_vnd,
    )
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
