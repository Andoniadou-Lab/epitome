#!/usr/bin/env python3
"""Daily smoke-test runner for epitome (server local time).

Example cron entry (runs at 03:00 every day):

    0 3 * * * cd /path/to/epitome/code && python run_scheduled_tests.py >> test_results/scheduled.log 2>&1
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from tests.run_tests import run  # noqa: E402


def main() -> int:
    return run(save=True)


if __name__ == "__main__":
    sys.exit(main())
