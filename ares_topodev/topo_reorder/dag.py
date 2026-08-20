"""Extract the ground-truth dependency DAG for a CaptainCookRecipes example.

This reads the *same* recipe JSON files that
`vendor/ares/src/exp_helpers/datasets/recipe_graph.py` (`RecipeGraphDatasetRaw`,
`RecipeGraphDatasetRaw2`, `RecipeGraphDatasetRaw3`) reads, but treats `edges` as
the ground-truth dependency graph rather than immediately collapsing it into one
fixed `topo_order`-based presentation.

Key finding (see plan / VENDOR.md): the recipe JSONs already contain an explicit
DAG (`steps`, `edges`) that is *independent* of the one stored `topo_order` the
vendored dataset loader happens to use. Nothing here needs to be reconstructed --
we just read it out directly instead of only reading the pre-linearized order.

We verified across all 24 recipes (see tests/test_dag.py) that:
  - there is exactly one node with text "START" and it has in-degree 0,
  - there is exactly one node with text "END" and it has out-degree 0.
So START/END are unambiguous anchors. `derived_claims` in the vendored code is
built from `topo_order[1:]`, i.e. START is never turned into a derived claim at
all (it's folded into context); END *is* a derived claim. `edges` alone does not
force END to be last (some recipes have steps with no path to END), so we pin it
last by convention -- see RecipeDag.position_bounds() below, which is a
documented, non-edge-derived modeling choice, not a structural necessity.
"""
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class RecipeDag:
    recipe_name: str
    steps: Dict[int, str]           # node id -> step text, ALL nodes (incl. START/END)
    edges: List[Tuple[int, int]]    # (u, v) meaning u must precede v; ALL edges as given
    start_id: int
    end_id: int
    # node ids that are actually permuted (i.e. derived_claims nodes: everything
    # except START, which vendored code never turns into a derived claim at all).
    # END *is* included here -- it is pinned last by topo_sample, not excluded.
    derived_node_ids: List[int]
    full_tgt2src: Dict[int, List[int]] = field(default_factory=dict)  # from ALL edges, for text
    full_src2tgt: Dict[int, List[int]] = field(default_factory=dict)

    def sortable_edges(self) -> List[Tuple[int, int]]:
        """Edges to respect when generating orderings of `derived_node_ids`.

        We keep every edge except ones touching START, since START is not part
        of the permuted sequence (it's always implicitly "before everything").
        Edges touching END are kept -- but see topo_sample.sample_orderings,
        which pins END last unconditionally, so those edges are automatically
        satisfied by construction; we keep them here anyway so
        assert_valid_topo_order can still check them uniformly.
        """
        return [(u, v) for (u, v) in self.edges if u != self.start_id]


def load_recipe_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_recipe_dag(data: dict, recipe_name: str) -> RecipeDag:
    steps = {int(k): v for k, v in data["steps"].items()}
    edges = [tuple(e) for e in data["edges"]]

    starts = [nid for nid, text in steps.items() if text == "START"]
    ends = [nid for nid, text in steps.items() if text == "END"]
    if len(starts) != 1:
        raise ValueError(f"{recipe_name}: expected exactly one START node, found {starts}")
    if len(ends) != 1:
        raise ValueError(f"{recipe_name}: expected exactly one END node, found {ends}")
    start_id, end_id = starts[0], ends[0]

    outdeg_end = sum(1 for (u, v) in edges if u == end_id)
    indeg_start = sum(1 for (u, v) in edges if v == start_id)
    if outdeg_end != 0:
        raise ValueError(f"{recipe_name}: END node has outgoing edges ({outdeg_end}); pinning it last would be unsound")
    if indeg_start != 0:
        raise ValueError(f"{recipe_name}: START node has incoming edges ({indeg_start})")

    full_tgt2src: Dict[int, List[int]] = defaultdict(list)
    full_src2tgt: Dict[int, List[int]] = defaultdict(list)
    for (u, v) in edges:
        full_src2tgt[u].append(v)
        full_tgt2src[v].append(u)

    derived_node_ids = [nid for nid in steps.keys() if nid != start_id]

    return RecipeDag(
        recipe_name=recipe_name,
        steps=steps,
        edges=edges,
        start_id=start_id,
        end_id=end_id,
        derived_node_ids=derived_node_ids,
        full_tgt2src=dict(full_tgt2src),
        full_src2tgt=dict(full_src2tgt),
    )


def load_all_recipe_dags(data_dir: str) -> Dict[str, RecipeDag]:
    dags = {}
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue
        recipe_name = filename[: -len(".json")]
        data = load_recipe_json(os.path.join(data_dir, filename))
        dags[recipe_name] = extract_recipe_dag(data, recipe_name)
    return dags
