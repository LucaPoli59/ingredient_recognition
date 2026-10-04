"""Reusable Phase 3 ingredient-learnability workflow.

Keep lightweight review utilities importable without loading the training stack.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingredient_selection.protocol import SelectorProtocol

__all__ = ["PROTOCOL_ID", "SelectorProtocol"]


def __getattr__(name):
    if name in __all__:
        from src.ingredient_selection import protocol
        return getattr(protocol, name)
    raise AttributeError(name)
