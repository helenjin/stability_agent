"""Minimal example: run SAGER (known-graph setting) on a toy diamond DAG.

    a -> b
    a -> c
    b -> d
    c -> d

Run with:  python -m sager.example
"""
import networkx as nx

from sager import MockEntailmentScorer, sager_known_graph


def build_toy_graph():
    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])

    claims = {
        "a": "We have flour and water.",
        "b": "Mixing flour and water makes dough.",
        "c": "Dough left to rise for an hour will double in size.",
        "d": "The risen dough can be baked into bread.",
    }
    base_priors = {"a": 0.95}  # only node "a" has no ancestors
    return G, claims, base_priors


def main():
    G, claims, base_priors = build_toy_graph()

    result = sager_known_graph(
        G,
        claims=claims,
        base_priors=base_priors,
        entailment_scorer=MockEntailmentScorer(),
        depth=None,  # unlimited ancestor depth (Anc_G(v))
        num_soundness_samples=200,
        max_orderings=10,
        seed=42,
        debug=False,
    )

    print("Graph-conditioned soundness estimates (tau_hat_G):")
    for node in nx.topological_sort(G):
        print(f"  {node}: {result.tau[node]:.4f}")

    print("\nDiagnostics:")
    for key, value in result.diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
