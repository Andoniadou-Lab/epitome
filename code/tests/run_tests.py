#!/usr/bin/env python3
"""Entry point for running the epitome smoke-test suite.

Invokes pytest on ``code/tests/`` programmatically, prints a colored summary,
optionally persists the latest result to ``code/test_results/latest.json``,
and returns the underlying pytest exit code so it can gate deployment.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pytest

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init()
    _GREEN = Fore.GREEN
    _RED = Fore.RED
    _YELLOW = Fore.YELLOW
    _CYAN = Fore.CYAN
    _RESET = Style.RESET_ALL
except ImportError:  # pragma: no cover - colorama is a soft dep
    _GREEN = _RED = _YELLOW = _CYAN = _RESET = ""


_TESTS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _TESTS_DIR.parent
_RESULTS_DIR = _CODE_DIR / "test_results"
_RESULTS_FILE = _RESULTS_DIR / "latest.json"


class TestSummary(TypedDict):
    passed: int
    failed: int
    skipped: int
    total: int


class _ResultCollector:
    """Pytest plugin that records the outcome of every test item."""

    def __init__(self) -> None:
        self.results: dict[str, str] = {}

    def pytest_runtest_logreport(self, report) -> None:  # noqa: D401
        if report.when != "call" and not (report.when == "setup" and report.outcome == "failed"):
            return
        previous = self.results.get(report.nodeid)
        if previous == "failed":
            return
        self.results[report.nodeid] = report.outcome


def _print_header(text: str) -> None:
    bar = "=" * (len(text) + 4)
    print(f"\n{_CYAN}{bar}\n| {text} |\n{bar}{_RESET}")


def _format_outcome(outcome: str) -> str:
    if outcome == "passed":
        return f"{_GREEN}PASS{_RESET}"
    if outcome == "failed":
        return f"{_RED}FAIL{_RESET}"
    if outcome == "skipped":
        return f"{_YELLOW}SKIP{_RESET}"
    return outcome.upper()


def save_results(summary: TestSummary) -> Path:
    """Write the most recent run to ``test_results/latest.json`` (server local time)."""
    now = datetime.now()
    payload = {
        "date": now.strftime("%Y/%m/%d"),
        "passed": summary["passed"],
        "total": summary["total"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
        "run_at": now.isoformat(timespec="seconds"),
    }
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    return _RESULTS_FILE


def run(save: bool = False) -> int:
    """Run the suite and return an exit code (0 = all passed)."""
    _print_header("Running Epitome Smoke Tests")

    collector = _ResultCollector()
    pytest_args = [
        "-q",
        "--no-header",
        "--import-mode=importlib",
        f"--rootdir={_TESTS_DIR}",
        str(_TESTS_DIR),
    ]
    exit_code = pytest.main(pytest_args, plugins=[collector])

    _print_header("Smoke Test Summary")

    if not collector.results:
        print(f"{_RED}No tests were collected.{_RESET}")
        return exit_code or 1

    passed = failed = skipped = 0
    for nodeid in sorted(collector.results):
        outcome = collector.results[nodeid]
        short = nodeid.replace(str(_TESTS_DIR) + os.sep, "").replace(str(_TESTS_DIR), "")
        print(f"  {_format_outcome(outcome)}  {short}")
        if outcome == "passed":
            passed += 1
        elif outcome == "failed":
            failed += 1
        elif outcome == "skipped":
            skipped += 1

    total = passed + failed + skipped
    summary: TestSummary = {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
    }
    print(
        f"\nTotal: {total}  "
        f"{_GREEN}Passed: {passed}{_RESET}  "
        f"{_RED}Failed: {failed}{_RESET}  "
        f"{_YELLOW}Skipped: {skipped}{_RESET}"
    )

    if save:
        results_path = save_results(summary)
        print(f"Saved results to {results_path}")

    if failed == 0 and exit_code == 0:
        print(f"{_GREEN}All smoke tests passed.{_RESET}")
    else:
        print(f"{_RED}Smoke tests reported failures - do not deploy.{_RESET}")

    return int(exit_code)


def main() -> int:
    if str(_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(_CODE_DIR))
    return run(save="--save" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
