"""Modular, reusable implementation of the archived SafetyPool policy."""

from __future__ import annotations

from typing import Optional, Sequence

from .constants import POLICY_NAME


__all__ = ["POLICY_NAME", "main"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run SafetyPool while keeping package import lightweight."""
    from .cli import main as run

    run(argv)
