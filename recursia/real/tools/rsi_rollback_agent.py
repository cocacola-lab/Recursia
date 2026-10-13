#!/usr/bin/env python3
"""Restore one agent's reusable learning to its previous independent Git version."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cap.agent.rsi.store import METHODS
from cap.agent.rsi.versions import manage_versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--agent', '--method', dest='method', choices=METHODS, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--list', action='store_true', help='show current version and its predecessors')
    group.add_argument('--init-only', action='store_true', help='import recorded learning and checkpoint without rollback')
    group.add_argument('--dry-run', action='store_true', help='preview rollback without writing files or Git')
    args = parser.parse_args(argv)
    action = 'list' if args.list else 'init' if args.init_only else 'rollback'
    try:
        result = manage_versions(args.campaign, args.method, action=action, dry_run=args.dry_run)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'Agent version operation failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
