#!/usr/bin/env python3
"""
scenario_runner.py – Run scripted conversation scenarios against the agent.

Usage:
    python scripts/scenario_runner.py [--scenarios tests/scenarios.json] [--real]

Default runs in mock mode (MOCK=1). Pass --real to use live APIs.
Prints PASS / FAIL for each scenario and an overall summary.

Exit code 0 if >= 10 of 12 scenarios pass; else 1.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCENARIOS_FILE = Path("tests/scenarios.json")


def _set_mock(val: bool) -> None:
    os.environ["MOCK"] = "1" if val else "0"
    # Force reload of config in already-imported modules
    import importlib
    import app.config as cfg
    importlib.reload(cfg)


async def run_scenario(scenario: dict) -> tuple[bool, str]:
    """Run a single scenario and return (passed, reason)."""
    # Import here so MOCK env var is already set
    from app import llm as llm_module, kb as kb_module
    from app.persona import build_system_prompt
    from app.tags import strip

    system_prompt = build_system_prompt()
    history: list[dict] = []
    checks_passed = []
    checks_failed = []

    for turn in scenario.get("turns", []):
        user_text = turn["user"]
        expected = turn.get("expect", {})

        history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": system_prompt}] + history

        full_reply = ""
        try:
            async for token in llm_module.stream(messages):
                full_reply += token
        except Exception as exc:
            return False, f"LLM error: {exc}"

        # Strip tags for text checks
        reply_text = strip(full_reply)
        history.append({"role": "assistant", "content": full_reply})

        # Check expectations
        if "contains_no" in expected:
            for bad in expected["contains_no"]:
                if re.search(bad, reply_text, re.IGNORECASE):
                    checks_failed.append(f"Found forbidden pattern {bad!r} in reply")
        if "contains" in expected:
            for good in expected["contains"]:
                if not re.search(good, reply_text, re.IGNORECASE):
                    checks_failed.append(f"Missing expected pattern {good!r} in reply")
        if "service_id" in expected:
            m = re.search(r"@@service=([^@]*)@@", full_reply)
            sid = m.group(1).strip() if m else "none"
            if sid != expected["service_id"]:
                checks_failed.append(
                    f"Expected service_id={expected['service_id']!r}, got {sid!r}"
                )
        if "no_invented_links" in expected and expected["no_invented_links"]:
            # Simple heuristic: no http(s) links outside the KB's known domains
            kb_entries = kb_module.load()
            known_sites = {e["official_site"] for e in kb_entries}
            links = re.findall(r"https?://[^\s)>\"']+", reply_text)
            for link in links:
                if not any(link.startswith(s) for s in known_sites):
                    checks_failed.append(f"Invented link detected: {link}")
        if "no_fees" in expected and expected["no_fees"]:
            if re.search(r"₹|\brupee|\d+\s*(rs|inr)\b", reply_text, re.IGNORECASE):
                checks_failed.append("Fee amount mentioned in reply")

    if checks_failed:
        return False, "; ".join(checks_failed)
    return True, "All checks passed"


async def main(scenarios_file: Path, use_real: bool) -> None:
    _set_mock(not use_real)

    if not scenarios_file.exists():
        print(f"❌ Scenarios file not found: {scenarios_file}")
        sys.exit(1)

    with scenarios_file.open(encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"Running {len(scenarios)} scenarios (MOCK={not use_real})…\n")
    passed = 0
    failed = 0

    for i, scenario in enumerate(scenarios):
        name = scenario.get("name", f"Scenario {i+1}")
        ok, reason = await run_scenario(scenario)
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} [{i+1}/{len(scenarios)}] {name}")
        if not ok:
            print(f"       Reason: {reason}")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    threshold = max(10, total - 2)  # allow 2 failures
    if passed >= threshold:
        print(f"✅ PASS (>= {threshold}/{total})")
        sys.exit(0)
    else:
        print(f"❌ FAIL (need >= {threshold}/{total})")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default=str(SCENARIOS_FILE))
    parser.add_argument("--real", action="store_true", help="Use real APIs")
    args = parser.parse_args()
    asyncio.run(main(Path(args.scenarios), args.real))

