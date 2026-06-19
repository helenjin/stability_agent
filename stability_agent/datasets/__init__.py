"""Dataset adapters.

Each adapter normalizes one corpus into canonical ``Example`` objects
(see ``stability_agent.core.schema``) via the shared ``BaseAdapter``.
"""

from .base import BaseAdapter
from .claimspy import ClaimSpyAdapter

__all__ = ["BaseAdapter", "ClaimSpyAdapter"]
