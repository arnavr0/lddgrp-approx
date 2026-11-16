"""Command-line interface for solving JSON LDdGRP instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model import LDdGRPInstance
from .solver import ALNSConfig, ALNSSolver, Evaluation


def evaluation_to_dict(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "objective": evaluation.objective,
        "delivery_order": list(evaluation.candidate.delivery_order),
        "edge_stages_zero_based": list(evaluation.candidate.edge_stages),
        "stages": [
            {
                "stage_zero_based": stage.stage,
                "start_node": stage.start_node,
                "end_node": stage.end_node,
                "required_edge_indices": list(stage.edge_indices),
                "base_cost": stage.base_cost,
                "load_multiplier": stage.multiplier,
                "landing_takeoff_cost": round(
                    stage.weighted_cost / stage.multiplier - stage.base_cost, 12
                )
                if stage.multiplier > 0
                else 0.0,
                "weighted_cost": stage.weighted_cost,
                "steps": [
                    {
                        "kind": step.kind,
                        "from_node": step.from_node,
                        "to_node": step.to_node,
                        "cost": step.cost,
                        "required_edge_index": step.edge_index,
                    }
                    for step in stage.steps
                ],
            }
            for stage in evaluation.stages
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a discrete LDdGRP JSON instance with ALNS.")
    parser.add_argument("instance", type=Path, help="Path to the LDdGRP JSON instance.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Wall-clock ALNS limit (default: 10).")
    parser.add_argument("--iterations", type=int, default=5_000, help="Maximum ALNS iterations (default: 5000).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--exact-stage-limit",
        type=int,
        default=10,
        help="Maximum stage-edge count for exact DP decoding (default: 10).",
    )
    parser.add_argument(
        "--large-stage-local-search-passes",
        type=int,
        default=12,
        help="2-opt/relocation passes after large-stage construction (default: 12).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the solution certificate JSON.")
    args = parser.parse_args()

    data = json.loads(args.instance.read_text(encoding="utf-8"))
    instance = LDdGRPInstance.from_dict(data)
    solver = ALNSSolver(
        instance,
        ALNSConfig(
            time_limit_seconds=args.seconds,
            max_iterations=args.iterations,
            exact_stage_limit=args.exact_stage_limit,
            seed=args.seed,
            large_stage_local_search_passes=args.large_stage_local_search_passes,
        ),
    )
    result = solver.solve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluation_to_dict(result), indent=2) + "\n", encoding="utf-8")
    print(f"objective={result.objective:.6f}")
    print(f"delivery_order={list(result.candidate.delivery_order)}")
    print(f"solution={args.output}")


if __name__ == "__main__":
    main()

