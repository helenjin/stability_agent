"""Experiment 2: minimal, single-edge dependency-violating orderings.

Experiment 1 asked "does a method's score change under valid, irrelevant
reorderings?" This asks the complementary question: "does violating a real
dependency move the score by *more* than that same ordinary valid-reordering
noise, or does it get lost in it?" A method that's meaningfully tracking the
true dependency structure should react more strongly to an actual broken
dependency than to a harmless relabeling of presentation order; if it
doesn't, that's a much sharper indictment than Experiment 1 alone.

Construction: start from one already-sampled *valid* ordering (from
topo_sample.sample_orderings) and swap two nodes u, v that are *adjacent* in
it (position(v) = position(u) + 1) where (u, v) is a real edge in the DAG.
Swapping adjacent elements changes nothing else's relative order, so exactly
one edge is violated -- a controlled, minimal, single-variable perturbation,
the same philosophy as the dataset's own single-ingredient-deletion error
injection.

Because claim text is a pure function of node id (see recipe_example.py),
node v's text still explicitly names u as an already-completed prerequisite
("Because we have completed all previous steps (<u's text>), ... we can now
do <v's text>") -- but after the swap, u no longer appears anywhere in v's
premise prefix. That's a concrete, textually-checkable inconsistency, unlike
Experiment 1's reorderings, where everything a claim's text mentions really
is available in premises (just via a different superset).
"""
from dataclasses import dataclass
from typing import List, Tuple

from .dag import RecipeDag


def find_adjacent_edges(dag: RecipeDag, ordering: List[int]) -> List[Tuple[int, int]]:
    """Edges (u, v) in dag.edges where v sits immediately after u in `ordering`."""
    position = {nid: i for i, nid in enumerate(ordering)}
    adjacent_edges = []
    for (u, v) in dag.edges:
        if u not in position or v not in position:
            continue  # e.g. edges touching START, which isn't part of `ordering`
        if position[v] == position[u] + 1:
            adjacent_edges.append((u, v))
    return adjacent_edges


def swap_adjacent(ordering: List[int], u: int, v: int) -> List[int]:
    """Returns a new ordering with u and v (must be adjacent, v immediately
    after u) swapped -- every other node's relative order is unchanged."""
    position = {nid: i for i, nid in enumerate(ordering)}
    if position[v] != position[u] + 1:
        raise ValueError(f"u={u}, v={v} are not adjacent in this ordering (positions {position[u]}, {position[v]})")
    new_ordering = list(ordering)
    iu, iv = position[u], position[v]
    new_ordering[iu], new_ordering[iv] = new_ordering[iv], new_ordering[iu]
    return new_ordering


def violated_edges(dag: RecipeDag, ordering: List[int]) -> List[Tuple[int, int]]:
    """Every edge in dag.edges that `ordering` violates (position[u] >= position[v]).
    START is treated as always before position 0, matching topo_sample.assert_valid_topo_order."""
    position = {nid: i for i, nid in enumerate(ordering)}
    position[dag.start_id] = -1
    violated = []
    for (u, v) in dag.edges:
        if position.get(u, -1) >= position.get(v, -1):
            violated.append((u, v))
    return violated


@dataclass
class InvalidOrdering:
    edge_violated: Tuple[int, int]      # the (u, v) edge this ordering was constructed to violate
    ordering: List[int]                 # the swapped, now-invalid ordering
    base_ordering: List[int]            # the valid ordering it was derived from
    all_violated_edges: List[Tuple[int, int]]  # sanity check: should be exactly [edge_violated]


def generate_invalid_orderings(dag: RecipeDag, base_ordering: List[int], max_cases: int = None) -> List[InvalidOrdering]:
    """One invalid ordering per adjacent edge found in `base_ordering` (a
    valid ordering), each violating exactly that edge and nothing else."""
    cases = []
    for (u, v) in find_adjacent_edges(dag, base_ordering):
        swapped = swap_adjacent(base_ordering, u, v)
        actual_violations = violated_edges(dag, swapped)
        cases.append(
            InvalidOrdering(
                edge_violated=(u, v),
                ordering=swapped,
                base_ordering=base_ordering,
                all_violated_edges=actual_violations,
            )
        )
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases
