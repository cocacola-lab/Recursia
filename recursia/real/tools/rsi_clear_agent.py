#!/usr/bin/env python3
"""Clear one RSI agent's programs, skills, snapshots, results and counters."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cap.agent.rsi.reset import clear_method
from cap.agent.rsi.store import METHODS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="RSI campaign directory")
    parser.add_argument("--agent", "--method", dest="method", choices=METHODS, required=True,
                        help="the single agent workspace to clear")
    parser.add_argument("--dry-run", action="store_true", help="list entries without clearing learning or outputs")
    args = parser.parse_args(argv)
    try:
        result = clear_method(args.campaign, args.method, dry_run=args.dry_run)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Clear failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
