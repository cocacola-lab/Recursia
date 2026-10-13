#!/usr/bin/env python3
"""Recursia FR3 entry point; method identifiers retain ASPIRE compatibility."""
import sys
from cap.agent.rsi.cli import main
if __name__ == "__main__":
    argv = sys.argv[1:]
    raise SystemExit(main(argv[1:] if argv[:1] == ["rsi"] else argv))
