from __future__ import annotations

from lddgrp.benchmarks import load_official_benchmark, read_published_results


def test_official_benchmark_parser_preserves_lddgrp_parameters_and_metric() -> None:
    data = __file__.replace("test_benchmarks.py", "data")
    from pathlib import Path

    instance_path = Path(data) / "LDdGRP4_4_2_1.dat"
    results_path = Path(data) / "official_results.txt"
    benchmark = load_official_benchmark(instance_path, read_published_results(results_path))

    assert benchmark.instance.depot == 0
    assert benchmark.instance.delivery_demands == {3: 2.0}
    assert benchmark.instance.required_edges[0].service_cost == 3.0
    assert benchmark.instance.curb_weight == 15.0
    assert benchmark.instance.landing_takeoff_cost == 125.0
    assert benchmark.node_labels_by_internal_index == (1, 2, 3, 4)
    assert benchmark.generic_delivery_costs == {4: 75.0}
    assert benchmark.instance.cost(0, 3) == 2.0
