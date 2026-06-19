"""Shared base for dataset adapters.

An adapter turns one corpus into canonical ``Example`` objects. Subclasses
implement ``iter_examples``; everything else (filtering, counting) is shared.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..core.schema import Example


class BaseAdapter(ABC):
    """Normalize a corpus into canonical Examples."""

    #: short registry name, e.g. "claimspy_v1"
    name: str = ""

    @abstractmethod
    def iter_examples(self) -> Iterator[Example]:
        """Yield one canonical Example per item in the corpus."""

    def examples(self) -> list[Example]:
        return list(self.iter_examples())

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_examples())
