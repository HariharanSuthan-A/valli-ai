#!/usr/bin/env python3
"""
check_kb.py – Validate data/knowledge.json.

Usage:
    python scripts/check_kb.py              # normal: warn on unverified
    python scripts/check_kb.py --strict     # fail on unverified entries

Exit code 0 on success, 1 on error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import kb as kb_module


def main():
    parser = argparse.ArgumentParser(description="Validate knowledge.json")
    parser.add_argument("--strict", action="store_true", help="Fail on verified=false")
    parser.add_argument("--file", default=None, help="Path to knowledge.json")
    args = parser.parse_args()

    path = Path(args.file) if args.file else kb_module.KNOWLEDGE_FILE
    print(f"Checking {path} …")

    try:
        with path.open(encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as exc:
        print(f"❌ Could not read file: {exc}")
        sys.exit(1)

    try:
        warnings = kb_module.validate(entries, strict=args.strict)
    except kb_module.KBError as exc:
        print(f"❌ Validation error: {exc}")
        sys.exit(1)

    if warnings:
        print(f"\n⚠  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"   • {w}")
    else:
        print("✅ No warnings.")

    print(f"\n✅ {len(entries)} entries validated successfully.")
    if args.strict:
        print("   Strict mode: all entries are verified=true.")
    else:
        unverified = sum(1 for e in entries if not e.get("verified", False))
        if unverified:
            print(f"   {unverified} entry/entries still need human verification (verified=false).")


if __name__ == "__main__":
    main()

