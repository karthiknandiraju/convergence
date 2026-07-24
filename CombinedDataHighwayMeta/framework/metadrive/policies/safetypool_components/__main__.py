"""Module entry point for ``python -m policies.safetypool_components``."""

from __future__ import annotations

import sys


if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
    from .configuration import build_parser

    build_parser().parse_args()

if __name__ == "__main__":
    from . import main

    main()
