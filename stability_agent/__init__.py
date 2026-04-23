"""Tools for computing and analyzing soft-stability outputs."""

from .analysis import AnalysisConfig, StabilityAnalyzer

__all__ = [
    "AnalysisConfig",
    "StabilityAnalyzer",
    "sample_alpha_pertbs",
    "soft_stability_rate",
]


def __getattr__(name: str):
    if name in {"sample_alpha_pertbs", "soft_stability_rate"}:
        from .soft_stability import sample_alpha_pertbs, soft_stability_rate

        return {
            "sample_alpha_pertbs": sample_alpha_pertbs,
            "soft_stability_rate": soft_stability_rate,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
