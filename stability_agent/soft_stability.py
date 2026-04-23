"""Soft-stability sampling and evaluation utilities.

This is the package-local version of the original MythBusters soft-stability
helper so new experiments do not need to import from the notebook source tree.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs.
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "stability_agent.soft_stability requires torch. Install with "
            "`pip install stability-agent[compute]` or install torch manually."
        )


def _no_grad(func):
    if torch is None:
        return func
    return torch.no_grad()(func)


def sample_alpha_pertbs(
    alpha: torch.LongTensor,
    radius: int,
    num_samples: int,
) -> torch.LongTensor:
    """Sample uniformly from alpha' >= alpha with at most ``radius`` flips."""

    _require_torch()
    original_shape = alpha.shape
    alpha = alpha.view(-1)
    samples = alpha.view(1, -1).repeat(num_samples, 1)

    zero_indices = torch.nonzero(alpha == 0, as_tuple=False).squeeze()
    if zero_indices.ndim == 0:
        zero_indices = zero_indices.view(1)
    num_zeros = zero_indices.numel()
    radius = min(radius, num_zeros)

    log_flip_probs = torch.tensor(
        [
            math.lgamma(num_zeros + 1)
            - math.lgamma(i + 1)
            - math.lgamma(num_zeros - i + 1)
            for i in range(radius + 1)
        ],
        dtype=torch.float32,
        device=alpha.device,
    )

    gumbel_noise = -torch.log(
        -torch.log(torch.rand(num_samples, radius + 1, device=alpha.device))
    )
    num_flips = torch.argmax(log_flip_probs.view(1, -1) + gumbel_noise, dim=-1)

    for i, flips in enumerate(num_flips.tolist()):
        if flips == 0:
            continue
        flip_inds = torch.randperm(num_zeros, device=alpha.device)[:flips]
        samples[i, zero_indices[flip_inds]] = 1

    return samples.view(num_samples, *original_shape).long()


@_no_grad
def soft_stability_rate(
    llm: Any,
    prompt: str,
    x: list[str],
    hypothesis: str,
    true_label: int,
    alpha: torch.LongTensor,
    radius: int,
    epsilon: float = 0.1,
    delta: float = 0.1,
    return_all: bool = False,
    max_tries: int = 3,
) -> torch.Tensor | dict[str, Any]:
    """Estimate the soft-stability rate for an LLM classifier.

    The classifier is evaluated on sampled supersets of the selected evidence
    mask ``alpha``. The returned rate is the probability that sampled
    perturbations keep the reference label.
    """

    _require_torch()
    num_samples = int(math.log(2 / delta) / (2 * (epsilon**2))) + 1
    all_alpha_pertbs = sample_alpha_pertbs(alpha, radius, num_samples)
    selected_claims_all = [
        [x[i] for i in torch.nonzero(row, as_tuple=True)[0].tolist()]
        for row in all_alpha_pertbs
    ]

    prompts = [
        prompt.format(selected_claims, hypothesis)
        for selected_claims in selected_claims_all
    ]
    results = llm.generate(prompts)

    labels = []
    for i, result in enumerate(results):
        answer = None
        for _ in range(max_tries):
            answer = parse_answer_label(result)
            if answer is not None:
                break
            result = llm.single_generate(prompt.format(selected_claims_all[i], hypothesis))
            results[i] = result
        labels.append(answer)

    labels_tensor = torch.tensor(
        [float("nan") if label is None else label for label in labels],
        dtype=torch.float32,
    )
    all_matches = labels_tensor == float(true_label)
    soft_stab_rate = all_matches.float().mean()

    if return_all:
        return {
            "soft_stability_rate": soft_stab_rate,
            "true_label": true_label,
            "alpha_pertbs": all_alpha_pertbs,
            "results": results,
            "labels": labels_tensor,
            "matches": all_matches,
        }
    return soft_stab_rate


def parse_answer_label(result: str) -> int | None:
    """Parse ``Answer: <int>`` from an LLM response."""

    marker = "Answer:"
    if marker not in result:
        return None
    tail = result.split(marker, 1)[1].strip()
    if not tail:
        return None
    first = tail.split()[0]
    try:
        return int(first)
    except ValueError:
        return None

