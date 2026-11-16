"""Input model and validation for the discrete LDdGRP."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping, Sequence


EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class RequiredEdge:
    """A required undirected edge, indexed by position in ``required_edges``."""

    u: int
    v: int
    service_cost: float

    def __post_init__(self) -> None:
        if self.u == self.v:
            raise ValueError("A required edge must have two distinct endpoints.")
        if self.service_cost < 0:
            raise ValueError("Required-edge service costs must be non-negative.")


@dataclass(frozen=True, slots=True)
class LDdGRPInstance:
    """The discrete, single-drone LDdGRP used in the source formulation.

    ``travel_cost`` is the complete undirected non-required-edge cost matrix.
    Each required edge is serviced exactly once by the decoder; its parallel
    non-required edge remains available for deadheading through the matrix.
    """

    depot: int
    travel_cost: tuple[tuple[float, ...], ...]
    delivery_demands: Mapping[int, float]
    required_edges: tuple[RequiredEdge, ...]
    curb_weight: float
    landing_takeoff_cost: float

    def __post_init__(self) -> None:
        n = len(self.travel_cost)
        if n == 0:
            raise ValueError("travel_cost cannot be empty.")
        if not 0 <= self.depot < n:
            raise ValueError("The depot is not a node in travel_cost.")
        if self.curb_weight < 0 or self.landing_takeoff_cost < 0:
            raise ValueError("Weight and landing/takeoff cost must be non-negative.")

        for row in self.travel_cost:
            if len(row) != n:
                raise ValueError("travel_cost must be square.")
        for i in range(n):
            if self.travel_cost[i][i] < -EPSILON:
                raise ValueError("travel_cost diagonal must be zero.")
            if not isclose(self.travel_cost[i][i], 0.0, abs_tol=EPSILON):
                raise ValueError("travel_cost diagonal must be zero.")
            for j in range(n):
                cost = self.travel_cost[i][j]
                if cost < -EPSILON:
                    raise ValueError("travel_cost values must be non-negative.")
                if not isclose(cost, self.travel_cost[j][i], abs_tol=EPSILON):
                    raise ValueError("travel_cost must be symmetric.")

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if self.travel_cost[i][k] > self.travel_cost[i][j] + self.travel_cost[j][k] + EPSILON:
                        raise ValueError("travel_cost must satisfy the triangle inequality.")

        seen_deliveries: set[int] = set()
        for node, demand in self.delivery_demands.items():
            if not isinstance(node, int) or not 0 <= node < n:
                raise ValueError(f"Delivery node {node!r} is not a valid node index.")
            if node == self.depot:
                raise ValueError("The depot cannot also be a delivery node.")
            if node in seen_deliveries:
                raise ValueError("Delivery nodes must be unique.")
            if demand < -EPSILON:
                raise ValueError("Delivery demands must be non-negative.")
            seen_deliveries.add(node)

        for edge in self.required_edges:
            if not (0 <= edge.u < n and 0 <= edge.v < n):
                raise ValueError("A required-edge endpoint is not a valid node index.")
            if edge.service_cost + EPSILON < self.cost(edge.u, edge.v):
                raise ValueError(
                    "A required-edge service cost must be at least its parallel "
                    "non-required (deadheading) cost."
                )

    @property
    def node_count(self) -> int:
        return len(self.travel_cost)

    @property
    def delivery_nodes(self) -> tuple[int, ...]:
        return tuple(self.delivery_demands)

    @property
    def total_demand(self) -> float:
        return sum(self.delivery_demands.values())

    @property
    def stage_count(self) -> int:
        return len(self.delivery_demands) + 1

    def cost(self, from_node: int, to_node: int) -> float:
        return self.travel_cost[from_node][to_node]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LDdGRPInstance":
        try:
            matrix = tuple(tuple(float(value) for value in row) for row in data["travel_cost"])
            demands = {int(node): float(demand) for node, demand in data["delivery_demands"].items()}
            edges = tuple(
                RequiredEdge(int(edge["u"]), int(edge["v"]), float(edge["service_cost"]))
                for edge in data["required_edges"]
            )
            return cls(
                depot=int(data["depot"]),
                travel_cost=matrix,
                delivery_demands=demands,
                required_edges=edges,
                curb_weight=float(data["curb_weight"]),
                landing_takeoff_cost=float(data["landing_takeoff_cost"]),
            )
        except KeyError as error:
            raise ValueError(f"Missing required input field: {error.args[0]}") from error


def instance_to_dict(instance: LDdGRPInstance) -> dict[str, Any]:
    """Return a JSON-serialisable representation, useful in experiments."""

    return {
        "depot": instance.depot,
        "travel_cost": [list(row) for row in instance.travel_cost],
        "delivery_demands": dict(instance.delivery_demands),
        "required_edges": [
            {"u": edge.u, "v": edge.v, "service_cost": edge.service_cost}
            for edge in instance.required_edges
        ],
        "curb_weight": instance.curb_weight,
        "landing_takeoff_cost": instance.landing_takeoff_cost,
    }
