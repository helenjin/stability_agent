"""Build the self-contained Recipe Dependency Explorer HTML page.

Extracts each of the 24 CaptainCookRecipes ground-truth DAGs (steps, edges,
direct parents, full ancestor sets) via the same `extract_recipe_dag` used
throughout ares_topodev, embeds it as a JSON blob into template.html, and
writes the result to recipe_dag_explorer.html -- open that file directly in
a browser, no server or build step needed.

The ancestor BFS below intentionally duplicates
`eval_harness.run_experiment.compute_ancestors_by_node_id` rather than
importing it: that module pulls in torch via eval_harness.sager, which this
tool has no other reason to depend on.

Usage:
    python -m ares_topodev.tools.recipe_dag_explorer.generate
"""
import json
import os
from collections import deque

from ares_topodev.topo_reorder.dag import extract_recipe_dag, load_recipe_json

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(TOOL_DIR)))
RECIPE_DATA_DIR = os.path.join(REPO_ROOT, "ares_topodev", "vendor", "ares", "data", "recipe_graphs")
TEMPLATE_PATH = os.path.join(TOOL_DIR, "template.html")
OUTPUT_PATH = os.path.join(TOOL_DIR, "recipe_dag_explorer.html")


def compute_ancestors_by_node_id(dag):
    ancestors = {}
    for nid in dag.derived_node_ids:
        seen = set()
        queue = deque(dag.full_tgt2src.get(nid, []))
        while queue:
            parent = queue.popleft()
            if parent == dag.start_id or parent in seen:
                continue
            seen.add(parent)
            queue.extend(dag.full_tgt2src.get(parent, []))
        ancestors[nid] = sorted(seen)
    return ancestors


def build_all_recipe_data(recipe_data_dir: str) -> dict:
    out = {}
    for filename in sorted(os.listdir(recipe_data_dir)):
        if not filename.endswith(".json"):
            continue
        name = filename[:-len(".json")]
        data = load_recipe_json(os.path.join(recipe_data_dir, filename))
        dag = extract_recipe_dag(data, name)
        ancestors_by_node_id = compute_ancestors_by_node_id(dag)

        parents_by_node_id = {nid: [] for nid in dag.derived_node_ids}
        for u, v in dag.sortable_edges():
            parents_by_node_id.setdefault(v, []).append(u)
        for nid in parents_by_node_id:
            parents_by_node_id[nid].sort()

        out[name] = {
            "steps": {str(k): v for k, v in dag.steps.items()},
            "start_id": dag.start_id,
            "end_id": dag.end_id,
            "derived_node_ids": dag.derived_node_ids,
            "edges": [[u, v] for u, v in dag.sortable_edges()],
            "parents_by_node_id": {str(k): v for k, v in parents_by_node_id.items()},
            "ancestors_by_node_id": {str(k): v for k, v in ancestors_by_node_id.items()},
        }
    return out


def main():
    data = build_all_recipe_data(RECIPE_DATA_DIR)
    template = open(TEMPLATE_PATH, "r").read()
    output = template.replace("__DATA_JSON__", json.dumps(data))
    with open(OUTPUT_PATH, "w") as f:
        f.write(output)
    print(f"wrote {OUTPUT_PATH} ({len(output)} bytes, {len(data)} recipes)")


if __name__ == "__main__":
    main()
