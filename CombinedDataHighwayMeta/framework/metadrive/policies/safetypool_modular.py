#!/usr/bin/env python3
"""Compatibility launcher for the modular SafetyPool implementation.

It accepts the same command-line options and writes the same
``Karthikeya27adv8`` policy folder as the archived monolithic implementation.
The help path deliberately avoids importing PyTorch or MetaDrive, which makes
configuration inspection possible on documentation-only systems.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch help cheaply and full experiments through the modular CLI."""
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        from safetypool_components.configuration import build_parser

        build_parser().parse_args()
        return

    from safetypool_components.cli import main as run

    run()


if __name__ == "__main__":
    main()

