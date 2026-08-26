"""SAGER (Structure-Aware Guarantees for Evaluating Reasoning) -- gold-graph
version. Replaces ARES's sequence-defined candidate-premise universe

    Pre_pi(c) = {p : p occurs before c in the presentation order pi}

with the graph-defined universe

    Pa_G(c) = {p : (p, c) is an edge in the gold dependency DAG G}

while reusing ARES's actual probabilistic-soundness machinery
(`stability_rate_deterministic`, `sample_s_pertbs` in
`exp_helpers.methods.utils.stability_deterministic`) as literally as possible.
Nothing here reimplements entailment scoring, the epsilon/delta -> N formula,
or the Monte Carlo resample-and-average estimator -- those are imported and
called unmodified. What's new is (a) a per-node *projection* of the running
weighted-sample population onto only the raw claims + a node's graph parents,
computed just before each `stability_rate_deterministic` call, and (b) a
*broadcast* of that call's resulting per-pattern soundness score back onto
every row of the FULL (unprojected) population before appending the new
column -- because a different node later in the graph may still need a
column this node's own projection dropped.

## Why the running population needs both a projection step and a broadcast step

ARES's own `tree_stability_rate_deterministic` never needs either step: every
claim's premises are *literally* "everything scored so far" (a growing
prefix), so the population it carries from one claim to the next already IS
the exact set of columns the next claim needs -- no projection required, and
appending the new column back onto that same population is trivially
correct (there is only one population, not a projected view of a larger one).

Once premises are graph parents instead of a prefix, a node's parent set can
be a non-contiguous subset of previously-scored nodes (e.g. in
`A->C, B->D, C,D->E`, D's premises = {B} must exclude A and C even though
both were already scored). There is no ARES code path for this -- it must be
designed, not lifted. The design used here:

  1. Maintain ONE running weighted-sample population per graph, `full_samples`
     / `full_counts`: `full_samples` is an [R, C] {0,1} tensor, one column
     per raw claim (fixed, added once) plus one column per graph node scored
     so far (added in traversal order); `full_counts` is its per-row weight.
     This is exactly ARES's own `samples`/`counts` state, just carrying more
     columns than any single node's own premises.
  2. To score node c: project `full_samples`/`full_counts` onto columns
     {raw claims} u Pa_G(c) (canonical order: raw claims in their fixed
     dataset order, then parents sorted ascending by node id -- see
     `CANONICAL_ORDER_RULE` below). Feed the projected (samples, counts)
     into the UNMODIFIED `stability_rate_deterministic` as its
     `samples=`/`counts=` kwargs with `exact=False`. That function always
     calls `sample_s_pertbs` in its existing-samples branch first, which
     resamples via `torch.multinomial` and dedupes via
     `.unique(dim=0, return_counts=True)` -- this is what merges any
     duplicate rows the projection created. No new merge logic is written;
     it's the same call ARES already makes for every non-root claim.
  3. That call returns, among the resampled/deduped *projected* rows, a
     per-row entailment score. Build a lookup: projected-pattern -> score.
     Broadcast this onto every row of the FULL (unprojected) population by
     re-computing each full row's own projected pattern (an exact sub-vector
     of that row, no approximation) and looking up its score. See
     "Coverage-gap fallback" below for the one case this lookup can miss.
  4. Append the new column onto the FULL population using `_append_column`,
     a direct port of `tree_stability_rate_deterministic`'s local
     `process_samples_and_counts` closure -- identical arithmetic, just
     decoupled from "the samples/y_pertbs must be this same call's own
     output" so it can operate on the broadcast values instead.
  5. Resample the FULL population back down to <=N rows via
     `sample_s_pertbs`'s existing-samples branch (same function, reused
     again) before moving to the next node. Without this, `full_samples`
     would double in row count every node (`_append_column` always doubles),
     growing unboundedly across a graph with many nodes -- ARES's own linear
     case never has this problem because the single carried-forward tensor
     already gets resampled-to-N as a side effect of being fed straight into
     the next claim's `stability_rate_deterministic` call. Here that
     resample is done explicitly, on the full (not just projected) tensor,
     since the two are no longer the same object.

## Canonical parent ordering (CANONICAL_ORDER_RULE)

Whenever more than one parent must be serialized into a premises list, they
are sorted by ascending integer node id. This is applied identically
regardless of which topological order was used to traverse the graph, so the
premises text list -- and therefore the prompt -- for a given node is a
function of (raw claims, node id, parent id set) only, never of traversal
order. This is what section 8 of the SAGER spec requires: the underlying
entailment model can itself be order-sensitive to premise serialization, so
serialization order must not silently reintroduce the sequence-sensitivity
the graph substitution is supposed to remove.

## Root / base claims

A node with `Pa_G(c) = {}` is scored using raw claims only -- the same
premise universe ARES's own first-in-sequence claim gets (`ki=0` in
`BaseDataset.get_data_entry`: `children[...] = sents_keys + all_hyps_keys[:0]
== sents_keys`). No new prior is invented for root claims; the existing
`stability_rate_deterministic` call path (raw claims run through
`sample_s_pertbs`'s fresh-Bernoulli branch) is reused exactly, whether that
root happens to be first in the traversal order or not. See
`graph_tree_stability_rate`'s `full_samples is None` branch (true only for literally the
first node scored in the whole run -- necessarily a graph root, since
`node_order` is required to be topological) vs. the general branch (a later
root just projects onto raw-claims-only columns).

## Coverage-gap fallback

`full_samples` is kept to <=N rows by the resample step in (5) above, so its
projection onto any Pa_G(c) subset has at most that many distinct patterns --
usually far fewer, since projecting merges rows. `stability_rate_deterministic`
draws N indices (with replacement) from that projected population before
querying the entailment model, so with N draws over <=N distinct patterns
every pattern is queried with high probability, but a low-weight pattern can
still fail to be drawn on a given step by chance. This is the same order of
Monte Carlo variance ARES's own linear case already has (a rare combination
can go un-redrawn on any step there too) -- not a new failure mode introduced
by projection. When a full-population row's own pattern was not among the
ones actually queried, `_broadcast_scores` assigns it the count-weighted
average score over the patterns that were queried, rather than raising or
silently guessing zero. This fallback is logged in each node's raw record
(`num_uncovered_rows`) so it is visible, not hidden, in the output; see the
certification-guarantee assessment in the SAGER deliverables writeup for what
this means for the epsilon/delta bound.

## A second, distinct finite-sample approximation: shared-population reweighting

Even with node-id-keyed seeding (see `_seed_for`), this design does NOT give
bit-exact topological invariance for a node scored via two different
traversal orders -- only approximate invariance, bounded by the same
Monte Carlo tolerance the epsilon/delta machinery already targets. Concrete
example: root node B (Pa_G(B) = {}) scored FIRST in one traversal draws a
fresh Bernoulli(p) sample directly over raw claims (the `full_samples is
None` branch). Scored AFTER some unrelated sibling A in another traversal,
B instead projects the *shared* population (whose row weights already
reflect A's own append-and-resample step) down to just the raw columns.
These two are equal in expectation -- appending and reweighting by A's score
then marginalizing A back out preserves each raw pattern's total weight, up
to `.ceil()` rounding -- but at finite N they are two different realized
distributions, not the same draw. This is a genuine trade-off of the
"shared population + projection" design relative to scoring every node from
independent, freshly-drawn marginals (which would give exact per-root
invariance but discard cross-branch correlation). It is a second,
independent source of approximation from the coverage-gap fallback above:
coverage gaps come from resampling a population down to <=N rows and
possibly missing a rare pattern; this comes from the population's own
marginal being only approximately, not exactly, preserved under
append-then-project. Both shrink as N grows (i.e. as epsilon shrinks), and
both are reported empirically via the TopoDev_SAGER(G) sanity check rather
than assumed to be zero.
"""
import hashlib
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ares_topodev.eval_harness import _bootstrap  # noqa: F401  (sys.path + OPENAI_API_KEY placeholder)
from exp_helpers.methods.base_scorer import BaseStabilityScorer, StabilityRateResults
from exp_helpers.methods.utils.stability_deterministic import sample_s_pertbs, stability_rate_deterministic


def _append_column(samples: torch.Tensor, counts: torch.Tensor, y: torch.Tensor):
    """Port of `tree_stability_rate_deterministic`'s local
    `process_samples_and_counts` closure (stability_deterministic.py), with
    one change: `samples`/`counts`/`y` are taken as explicit arguments
    instead of being read off `stab_rate_results` for this same call, so the
    scores used to extend a row can come from a *different* (projected)
    computation than the row itself. The arithmetic -- the two-way split
    into a "retained" and "not retained" copy of each row, the min-nonzero
    rescaling, the ceil-and-clamp-to-at-least-1 count adjustment -- is
    unchanged from the original.
    """
    samples_with_1 = torch.cat([samples, torch.ones_like(y)[:, None]], dim=-1)
    samples_with_0 = torch.cat([samples, torch.zeros_like(y)[:, None]], dim=-1)
    new_samples = torch.cat([samples_with_1, samples_with_0], dim=0)

    y_probs = torch.stack([y, 1 - y], dim=1)
    nonzero = y_probs[y_probs > 0]
    min_nonzero = nonzero.min().item() if nonzero.numel() > 0 else 1.0
    scale_factor = 1.0 / min_nonzero if min_nonzero < 1 else 1.0

    y_scaled = y * scale_factor
    one_minus_y_scaled = (1 - y) * scale_factor

    counts_with_1 = (counts * y_scaled).ceil().long()
    counts_with_0 = (counts * one_minus_y_scaled).ceil().long()
    counts_with_1 = torch.maximum(counts_with_1, (counts > 0).long() * (y > 0).long())
    counts_with_0 = torch.maximum(counts_with_0, (counts > 0).long() * ((1 - y) > 0).long())

    new_counts = torch.cat([counts_with_1, counts_with_0], dim=0)
    return new_samples, new_counts


_RNG_LOCK = threading.Lock()


def _locked_sample_s_pertbs(seed_value: int, *args, **kwargs):
    """Reseed torch's global RNG and immediately call `sample_s_pertbs`,
    atomically. `torch.manual_seed` sets *process-global* state, and
    `run_experiment.py`'s `recipe_concurrency` runs multiple recipes'
    scoring concurrently in separate threads, all sharing this process's one
    RNG. Without a lock, thread A's `manual_seed(x)` can land between thread
    B's `manual_seed(y)` and B's actual `torch.rand`/`torch.multinomial`
    draw -- silently defeating the node-id-keyed "common random numbers"
    scheme `_seed_for` exists to provide, with no error or symptom other
    than results quietly not being what a fixed seed would predict.

    Deliberately scoped to just this fast, CPU-only call, not the slower
    network-bound entailment scoring that follows it in
    `graph_tree_stability_rate` -- that part stays unlocked, so
    `recipe_concurrency`'s actual benefit (overlapping API latency across
    recipes) is preserved. This is why `graph_tree_stability_rate` always
    pre-computes its samples here and hands them to
    `stability_rate_deterministic` with `exact=True` (skip re-sampling,
    use what I already gave you) rather than letting that function call
    `sample_s_pertbs` internally, unlocked, on its own.
    """
    with _RNG_LOCK:
        torch.manual_seed(seed_value)
        return sample_s_pertbs(*args, **kwargs)


def _resample_bounded(samples: torch.Tensor, counts: torch.Tensor, num_samples: int, seed_value: int):
    """Resample <=num_samples unique rows from (samples, counts), weighted by
    counts -- reuses `sample_s_pertbs`'s existing-samples branch verbatim
    (the same multinomial-then-unique step ARES already runs on its carried
    state before scoring every claim); the dummy `s` argument is unused on
    this branch (only its shape matters, and only when existing_samples is
    None, which it never is here)."""
    dummy_s = torch.ones(samples.shape[1])
    resampled, new_counts, _, _ = _locked_sample_s_pertbs(
        seed_value, dummy_s, p=1.0, num_samples=num_samples, exact=False, existing_samples=samples, existing_counts=counts
    )
    return resampled, new_counts


def _seed_for(seed: int, node_id, salt: str) -> int:
    """Deterministic seed keyed by (run seed, node id, salt) -- NOT by
    traversal position or call order. Re-seeding torch's global RNG with this
    immediately before each stochastic step (the fresh-Bernoulli/resample
    calls inside `stability_rate_deterministic` and `_resample_bounded`)
    means a given node's random draws depend only on which node it is, never
    on what order the graph happened to be traversed in -- the "common
    random numbers" technique section 12 of the SAGER spec calls for, used
    to separate genuine algorithmic non-invariance from ordinary Monte Carlo
    noise when checking TopoDev_SAGER(G) == 0. This does not, by itself,
    guarantee bit-exact cross-order equality (see module docstring's
    discussion of ceil()-rounding and finite-N resampling noise); it removes
    *unseeded* run-to-run/order-to-order noise as a confound, leaving only
    that residual, epsilon-bounded Monte Carlo variance.
    """
    digest = hashlib.sha256(f"sager|{seed}|{node_id}|{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _broadcast_scores(full_view: torch.Tensor, query_samples: torch.Tensor, query_y: torch.Tensor, query_counts: torch.Tensor):
    """For every row of `full_view` (a projection of the full population
    onto the same columns `query_samples` is over), look up the score
    computed for its exact pattern. A pattern present in `full_view` but not
    among `query_samples` (see "Coverage-gap fallback" in the module
    docstring) gets the count-weighted average score over the patterns that
    were queried. Returns (scores, num_uncovered)."""
    lookup: Dict[tuple, float] = {}
    for row, y in zip(query_samples.tolist(), query_y.tolist()):
        lookup[tuple(row)] = y

    total_weight = query_counts.sum().item()
    fallback = (query_y * query_counts.float()).sum().item() / total_weight if total_weight > 0 else 0.5

    scores = []
    num_uncovered = 0
    for row in full_view.tolist():
        key = tuple(row)
        if key in lookup:
            scores.append(lookup[key])
        else:
            scores.append(fallback)
            num_uncovered += 1
    return torch.tensor(scores, dtype=torch.float), num_uncovered


@dataclass
class GraphNodeRecord:
    node_id: int
    parent_ids: List[int]
    premises_text: List[str]
    hypothesis_text: str
    stability_rate: float
    N: int
    seed: int
    num_uncovered_rows: int
    # The queried weighted-sample population for THIS node, in the same
    # column order as premises_text (query_samples[r][i] = 1 iff premises_text[i]
    # was retained in sampled row r). This is data stability_rate_deterministic
    # already computes to reach `stability_rate` -- previously discarded once
    # aggregated. Persisting it lets a later analysis attribute how much any
    # single premise (e.g. one specific ancestor, for sager_ancestors) moved
    # the score, by correlating that premise's retention bit against query_y
    # across these rows -- at ZERO additional model calls, since this is
    # exactly the data already paid for scoring the node once, not a new
    # leave-one-out ablation (which needs k+1 fresh full node-scorings per
    # node and does not scale to a full dataset -- see conversation).
    query_samples: List[List[int]]
    query_y: List[float]
    query_counts: List[int]


def graph_tree_stability_rate(
    entailment_model,
    node_order: List[int],
    parents_by_node_id: Dict[int, List[int]],
    text_by_node_id: Dict[int, str],
    raw_claims: List[str],
    p: float = 0.95,
    epsilon: float = 0.1,
    delta: float = 0.1,
    entailment_mode: str = "granular",
    temperature: float = 0.0,
    seed: int = 42,
):
    """Graph-conditioned counterpart of `tree_stability_rate_deterministic`.
    `node_order` is a valid topological order of the graph, used ONLY to
    decide a legal computation sequence (parents must be scored before
    children) -- it is never used to construct a node's premise set. Returns
    (stability_rates: List[float] aligned to node_order, records: List[GraphNodeRecord]).

    `seed` is combined with each node's own id (never with its position in
    `node_order`) to reseed torch's global RNG immediately before every
    stochastic step for that node -- see `_seed_for`. Passing the same `seed`
    across two runs that use different (but both valid) topological orders
    is exactly the "controlled random seeds / common random numbers" setup
    section 12 calls for.
    """
    m = len(node_order)
    n_raw = len(raw_claims)
    N = math.ceil(math.log(2 * m / delta) / (2 * (epsilon**2)))  # same union-bound formula as tree_stability_rate_deterministic, m = |V|

    raw_labeled = [f"raw{ri + 1}: {text}" for ri, text in enumerate(raw_claims)]

    full_samples: Optional[torch.Tensor] = None
    full_counts: Optional[torch.Tensor] = None
    col_ids: List[str] = []
    node_col_idx: Dict[int, int] = {}

    stability_rates: List[float] = []
    records: List[GraphNodeRecord] = []

    for node_id in node_order:
        parent_ids = sorted(parents_by_node_id.get(node_id, []))  # CANONICAL_ORDER_RULE: ascending node id
        hyp_text = f"step{node_id}: {text_by_node_id[node_id]}"

        if full_samples is None:
            if parent_ids:
                raise ValueError(
                    f"node {node_id} is first in node_order but has parents {parent_ids} -- "
                    "node_order must be a valid topological order of the graph"
                )
            premises = raw_labeled
            pre_samples, pre_counts, _, _ = _locked_sample_s_pertbs(
                _seed_for(seed, node_id, "score"), torch.ones(n_raw), p=p, num_samples=N, exact=False
            )
            # exact=True here means "use the samples/counts I'm handing you
            # as-is, don't call sample_s_pertbs yourself" -- NOT literally
            # exhaustive-subset enumeration (that's sample_s_pertbs's own,
            # differently-scoped `exact` flag, which we already resolved
            # above under the lock). This is what lets the network-bound
            # entailment scoring below run unlocked/concurrently: all RNG
            # consumption for this node already happened, atomically, above.
            stab = stability_rate_deterministic(
                entailment_model,
                premises,
                hyp_text,
                p=p,
                epsilon=epsilon,
                delta=delta,
                samples=pre_samples,
                counts=pre_counts,
                exact=True,
                entailment_mode=entailment_mode,
                N=N,
                num_raw=0,
                temperature=temperature,
                return_all=True,
            )
            col_ids = [f"raw{ri + 1}" for ri in range(n_raw)]
            base_samples = torch.tensor(stab["samples"], dtype=torch.long)
            base_counts = torch.tensor(stab["counts"], dtype=torch.long)
            y_for_append = torch.tensor(stab["y_pertbs"], dtype=torch.float)
            full_samples, full_counts = _append_column(base_samples, base_counts, y_for_append)
            num_uncovered = 0
        else:
            raw_col_idxs = list(range(n_raw))  # raw columns are always the fixed first n_raw columns, in order
            parent_col_idxs = [node_col_idx[pid] for pid in parent_ids]
            proj_col_idxs = raw_col_idxs + parent_col_idxs
            proj_samples = full_samples[:, proj_col_idxs]
            proj_counts = full_counts
            premises = raw_labeled + [f"step{pid}: {text_by_node_id[pid]}" for pid in parent_ids]

            pre_samples, pre_counts, _, _ = _locked_sample_s_pertbs(
                _seed_for(seed, node_id, "score"),
                torch.ones(len(proj_col_idxs)),
                p=p,
                num_samples=N,
                exact=False,
                existing_samples=proj_samples,
                existing_counts=proj_counts,
            )
            stab = stability_rate_deterministic(  # exact=True: see bootstrap branch's comment above
                entailment_model,
                premises,
                hyp_text,
                p=p,
                epsilon=epsilon,
                delta=delta,
                samples=pre_samples,
                counts=pre_counts,
                exact=True,
                entailment_mode=entailment_mode,
                N=N,
                num_raw=0,
                temperature=temperature,
                return_all=True,
            )
            query_samples = torch.tensor(stab["samples"], dtype=torch.long)
            query_y = torch.tensor(stab["y_pertbs"], dtype=torch.float)
            query_counts = torch.tensor(stab["counts"], dtype=torch.long)

            full_proj_view = full_samples[:, proj_col_idxs]
            y_broadcast, num_uncovered = _broadcast_scores(full_proj_view, query_samples, query_y, query_counts)
            full_samples, full_counts = _append_column(full_samples, full_counts, y_broadcast)

        full_samples, full_counts = _resample_bounded(
            full_samples, full_counts, N, _seed_for(seed, node_id, "resample")
        )

        col_ids.append(f"step{node_id}")
        node_col_idx[node_id] = len(col_ids) - 1

        stability_rates.append(stab["stability_rate"])
        records.append(
            GraphNodeRecord(
                node_id=node_id,
                parent_ids=parent_ids,
                premises_text=premises,
                hypothesis_text=hyp_text,
                stability_rate=stab["stability_rate"],
                N=N,
                seed=seed,
                num_uncovered_rows=num_uncovered,
                # stab["samples"]/["y_pertbs"]/["counts"] are already plain
                # Python lists (stability_rate_deterministic's own
                # return_all=True branch calls .tolist()) -- same column
                # order as `premises` above.
                query_samples=stab["samples"],
                query_y=stab["y_pertbs"],
                query_counts=stab["counts"],
            )
        )

    return stability_rates, records


class SagerStabilityScorer(BaseStabilityScorer):
    """Gold-graph SAGER: same constructor surface as
    `exp_helpers.methods.cert_nonexact.CertNonexactStabilityScorer` (p,
    epsilon, delta, entailment_mode, temperature), reusing
    `stability_rate_deterministic` for every individual entailment
    computation. The only thing that changes relative to that scorer is
    which premises are supplied per node -- see module docstring.

    `input_dict` schema (see `build_graph_data_entry`):
      {
        'node_order': [node_id, ...],            # valid topological order; traversal only
        'raw_claims': [text, ...],
        'text_by_node_id': {node_id: text},
        'parents_by_node_id': {node_id: [parent_node_id, ...]},
      }
    """

    def __init__(
        self,
        entailment_model,
        p: float = 0.95,
        epsilon: float = 0.1,
        delta: float = 0.1,
        entailment_mode: str = "granular",
        temperature: float = 0.0,
        seed: int = 42,
    ):
        self.entailment_model = entailment_model
        self.p = p
        self.epsilon = epsilon
        self.delta = delta
        self.entailment_mode = entailment_mode
        self.temperature = temperature
        self.seed = seed

    def get_stability_rate(self, input_dict: Dict) -> StabilityRateResults:
        stability_rates, records = graph_tree_stability_rate(
            self.entailment_model,
            node_order=input_dict["node_order"],
            parents_by_node_id=input_dict["parents_by_node_id"],
            text_by_node_id=input_dict["text_by_node_id"],
            raw_claims=input_dict["raw_claims"],
            p=self.p,
            epsilon=self.epsilon,
            delta=self.delta,
            entailment_mode=self.entailment_mode,
            temperature=self.temperature,
            seed=self.seed,
        )
        stability_rate = stability_rates[-1] if stability_rates else None
        return StabilityRateResults(
            stability_rate=stability_rate,
            stability_rates=stability_rates,
            inputs=[
                {
                    "node_id": r.node_id,
                    "parent_ids": r.parent_ids,
                    "premises_text": r.premises_text,
                    "hypothesis_text": r.hypothesis_text,
                }
                for r in records
            ],
            children=None,
            parents=input_dict["parents_by_node_id"],
            stab_rate_results=[
                {
                    "node_id": r.node_id,
                    "parent_ids": r.parent_ids,
                    "premises_text": r.premises_text,  # same column order as query_samples below
                    "stability_rate": r.stability_rate,
                    "N": r.N,
                    "seed": r.seed,
                    "num_uncovered_rows": r.num_uncovered_rows,
                    "query_samples": r.query_samples,
                    "query_y": r.query_y,
                    "query_counts": r.query_counts,
                }
                for r in records
            ],
        )


def build_graph_data_entry(
    raw_claims: List[str],
    node_order: List[int],
    text_by_node_id: Dict[int, str],
    parents_by_node_id: Dict[int, List[int]],
) -> Dict:
    """Graph-aware counterpart of `exp_helpers.datasets.base.BaseDataset.get_data_entry`.
    `node_order` fixes a computation sequence only (must be a valid
    topological order of the graph); `parents_by_node_id` -- not position in
    `node_order` -- determines each node's premises."""
    return {
        "node_order": node_order,
        "raw_claims": raw_claims,
        "text_by_node_id": text_by_node_id,
        "parents_by_node_id": parents_by_node_id,
    }
