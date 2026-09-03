"""Scratch validation script -- Monte Carlo convergence check for SAGER
(known dependency graph). Two distinct things checked, per the two failure
modes discussed:

  (1) Variance shrinkage: does the spread of tau_hat_G(c) across independent
      seeds shrink as N grows, roughly like 1/sqrt(N)?
  (2) Bias vanishing: does the mean of tau_hat_G(c) across seeds actually
      approach an INDEPENDENTLY, ANALYTICALLY computed expected value as N
      grows (not just "look stable")?

Uses a>c, b>c with an order-insensitive constant-per-subset mock scorer so
the true expectation E[p_c] can be computed by hand from p_a, p_b and the
four possible premise subsets.
"""
import math
import statistics

import networkx as nx

from sager import sager_known_graph

G = nx.DiGraph()
G.add_nodes_from(["a", "b", "c"])
G.add_edges_from([("a", "c"), ("b", "c")])
claims = {"a": "Claim A", "b": "Claim B", "c": "Claim C"}

P_A, P_B = 0.3, 0.7
SCORES = {
    frozenset(): 0.2,
    frozenset(["Claim A"]): 0.5,
    frozenset(["Claim B"]): 0.6,
    frozenset(["Claim A", "Claim B"]): 0.9,
}


def order_insensitive_scorer(premises, claim):
    return SCORES[frozenset(premises)]


true_e_p_c = (
    (1 - P_A) * (1 - P_B) * SCORES[frozenset()]
    + P_A * (1 - P_B) * SCORES[frozenset(["Claim A"])]
    + (1 - P_A) * P_B * SCORES[frozenset(["Claim B"])]
    + P_A * P_B * SCORES[frozenset(["Claim A", "Claim B"])]
)
print("=" * 70)
print("Analytic expectation (independent of SAGER's implementation)")
print("=" * 70)
print(f"  P(A=empty)   = {(1-P_A)*(1-P_B):.4f}  x  E=0.2  = {(1-P_A)*(1-P_B)*0.2:.4f}")
print(f"  P(A={{a}})     = {P_A*(1-P_B):.4f}  x  E=0.5  = {P_A*(1-P_B)*0.5:.4f}")
print(f"  P(A={{b}})     = {(1-P_A)*P_B:.4f}  x  E=0.6  = {(1-P_A)*P_B*0.6:.4f}")
print(f"  P(A={{a,b}})   = {P_A*P_B:.4f}  x  E=0.9  = {P_A*P_B*0.9:.4f}")
print(f"  E[p_c] = {true_e_p_c:.6f}")

print()
print("=" * 70)
print("1. Bias and variance across independent seeds, as N grows")
print("=" * 70)
Ns = [10, 50, 200, 1000, 5000]
NUM_SEEDS = 20
rows = []
for N in Ns:
    tau_c_values = []
    tau_a_values = []
    tau_b_values = []
    for seed in range(NUM_SEEDS):
        result = sager_known_graph(
            G, claims=claims, base_priors={"a": P_A, "b": P_B},
            entailment_scorer=order_insensitive_scorer,
            depth=math.inf, num_soundness_samples=N, max_orderings=10, seed=1000 + seed,
        )
        tau_c_values.append(result.tau["c"])
        tau_a_values.append(result.tau["a"])
        tau_b_values.append(result.tau["b"])

    # Base nodes: exact (up to float rounding) regardless of N (p_v^(i) = p_v every sample, no MC estimation).
    assert all(math.isclose(v, P_A, abs_tol=1e-9) for v in tau_a_values), tau_a_values
    assert all(math.isclose(v, P_B, abs_tol=1e-9) for v in tau_b_values), tau_b_values

    mean_c = statistics.mean(tau_c_values)
    std_c = statistics.pstdev(tau_c_values)
    bias = abs(mean_c - true_e_p_c)
    scaled_std = std_c * math.sqrt(N)  # should be roughly constant if std ~ C/sqrt(N)
    rows.append((N, mean_c, bias, std_c, scaled_std))
    print(f"  N={N:>5}: mean(tau_c)={mean_c:.4f}  |bias|={bias:.4f}  std(tau_c)={std_c:.4f}  std*sqrt(N)={scaled_std:.4f}")

print()
print("=" * 70)
print("Checks")
print("=" * 70)
biases = [r[2] for r in rows]
stds = [r[3] for r in rows]
scaled_stds = [r[4] for r in rows]

bias_shrinks = all(biases[i + 1] <= biases[i] * 1.5 for i in range(len(biases) - 1)) or biases[-1] < 0.02
print(f"  bias sequence: {[round(b, 4) for b in biases]}")
print(f"  bias at largest N ({Ns[-1]}) is small (<0.02): {biases[-1] < 0.02}")

std_shrinks = all(stds[i + 1] < stds[i] for i in range(len(stds) - 1))
print(f"  std sequence (should be monotonically decreasing): {[round(s, 4) for s in stds]}")
print(f"  std monotonically decreasing with N: {std_shrinks}")

scaled_std_mean = statistics.mean(scaled_stds)
scaled_std_relative_spread = (max(scaled_stds) - min(scaled_stds)) / scaled_std_mean
print(f"  std*sqrt(N) sequence (should be roughly constant under 1/sqrt(N) law): {[round(s, 4) for s in scaled_stds]}")
print(f"  relative spread of std*sqrt(N) across all N: {scaled_std_relative_spread:.3f}  (small => consistent with 1/sqrt(N) scaling)")

print()
print("=" * 70)
print("2. Running average within a single large-N run (does it stabilize, not drift?)")
print("=" * 70)
N_big = 5000
result = sager_known_graph(
    G, claims=claims, base_priors={"a": P_A, "b": P_B},
    entailment_scorer=order_insensitive_scorer,
    depth=math.inf, num_soundness_samples=N_big, max_orderings=10, seed=42, debug=True,
)
p_c_trace = [sample["c"]["p"] for sample in result.debug_samples]
checkpoints = [500, 1000, 2000, 3000, 4000, 5000]
running_avgs = []
for cp in checkpoints:
    avg = sum(p_c_trace[:cp]) / cp
    running_avgs.append(avg)
    print(f"  after {cp:>5} samples: running average of p_c = {avg:.4f}  (true E[p_c]={true_e_p_c:.4f}, diff={abs(avg-true_e_p_c):.4f})")

late_diffs = [abs(a - true_e_p_c) for a in running_avgs[-3:]]
early_diff = abs(running_avgs[0] - true_e_p_c)
stabilizing = max(late_diffs) <= early_diff or max(late_diffs) < 0.02
print(f"  final tau_c (N={N_big}): {result.tau['c']:.4f}")
print(f"  running average stabilizes toward true value (not drifting away): {stabilizing}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  True E[p_c] (analytic, independent of implementation): {true_e_p_c:.4f}")
print(f"  Bias shrinks with N: {bias_shrinks}")
print(f"  Std shrinks with N: {std_shrinks}")
print(f"  Std scaling consistent with 1/sqrt(N) (relative spread={scaled_std_relative_spread:.3f}): {scaled_std_relative_spread < 0.5}")
print(f"  Running average stabilizes near true value within a single run: {stabilizing}")
all_pass = bias_shrinks and std_shrinks and (scaled_std_relative_spread < 0.5) and stabilizing
print(f"  Overall convergence test: {'PASSED' if all_pass else 'FAILED'}")
