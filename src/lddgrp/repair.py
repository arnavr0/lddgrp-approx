"""Bounded exact repair primitives for the stage-allocation decisions.

The public ``EdgeStageRepairOracle`` protocol is deliberately solver agnostic.
An implementation backed by the paper's branch-and-cut formulation can replace
the enumeration below without changing the ALNS outer loop.  The included
implementation is an exact enumerator *for its stated neighbourhood*; it is
not a claim of global optimality and does not require a commercial MIP solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import monotonic
from typing import TYPE_CHECKING, Callable, Protocol

from .model import EPSILON

if TYPE_CHECKING:
    from .solver import Candidate, Evaluation


class EdgeStageRepairOracle(Protocol):
    """Improve a candidate while only changing selected edge-stage labels.

    ``mutable_edge_indices`` specifies the destroyed part of the solution and
    ``allowed_stages`` its permitted stage labels.  The delivery order and all
    other edge labels are fixed.  Implementations must return a feasible
    candidate and honor ``deadline`` as far as their backend permits.
    """

    def repair(
        self,
        candidate: Candidate,
        evaluate: Callable[[Candidate], Evaluation],
        *,
        mutable_edge_indices: tuple[int, ...],
        allowed_stages: tuple[int, ...],
        deadline: float,
    ) -> Candidate:
        """Return a repaired candidate, retaining the incumbent if necessary."""


@dataclass(frozen=True, slots=True)
class EnumerativeEdgeStageRepair:
    """Exact enumeration of a small edge-to-stage reassignment neighbourhood.

    At most ``max_mutable_edges`` labels are free.  The solver normally gives
    two adjacent stages, so the exhaustive search has at most
    ``2 ** max_mutable_edges`` candidates.  When the clock expires the best
    feasible incumbent seen so far is returned; then the search is no longer an
    exact solution of the local neighbourhood, but never returns an invalid
    candidate.
    """

    max_mutable_edges: int = 7

    def __post_init__(self) -> None:
        if self.max_mutable_edges < 1:
            raise ValueError("max_mutable_edges must be at least one.")

    def repair(
        self,
        candidate: Candidate,
        evaluate: Callable[[Candidate], Evaluation],
        *,
        mutable_edge_indices: tuple[int, ...],
        allowed_stages: tuple[int, ...],
        deadline: float,
    ) -> Candidate:
        stages = tuple(dict.fromkeys(allowed_stages))
        indices = tuple(dict.fromkeys(mutable_edge_indices))[: self.max_mutable_edges]
        if not indices or not stages or monotonic() >= deadline:
            return candidate

        incumbent = evaluate(candidate)
        best = incumbent
        for assignment in product(stages, repeat=len(indices)):
            if monotonic() >= deadline:
                break
            edge_stages = list(candidate.edge_stages)
            for edge_index, stage in zip(indices, assignment, strict=True):
                edge_stages[edge_index] = stage
            proposal = type(candidate)(candidate.delivery_order, tuple(edge_stages))
            proposal_eval = evaluate(proposal)
            if proposal_eval.objective + EPSILON < best.objective:
                best = proposal_eval
        return best.candidate
