"""Shared handling of "is wqio available, and does that have to matter?".

The tools under `tools/` compare this port against wqio. All three share the
same dilemma: wqio may simply not be installed, and the right answer differs by
caller.

A developer running a checker locally does not care that it skipped; a CI gate
that reports success while comparing nothing is worse than useless, because it
looks like evidence. So a missing wqio is a skip by default, and a hard failure
when `--require` is passed or `POLARS_ROS_REQUIRE_WQIO` is set. CI sets that
variable, so the gate can never pass vacuously.
"""

from __future__ import annotations

import argparse
import os

ENV_VAR = "POLARS_ROS_REQUIRE_WQIO"

_TRUTHY = {"1", "true", "yes", "on"}


def required_from_env() -> bool:
    """Whether the environment demands a real comparison."""
    return os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY


def add_require_flag(parser: argparse.ArgumentParser) -> None:
    """Register the --require flag these tools share."""
    parser.add_argument(
        "--require",
        action="store_true",
        help=f"fail instead of skipping when wqio is unavailable (also via {ENV_VAR})",
    )


def gate(require: bool, reason: str, install_hint: str = "") -> int | None:
    """Return an exit code if the caller should stop, else None to continue.

    Prints a SKIP notice, or a FAIL when `require` is set, so the reason is
    always visible in the output.
    """
    message = reason if not install_hint else f"{reason}; {install_hint}"
    if require:
        print(f"FAIL: {message}")
        return 1
    print(f"SKIP: {message}")
    return 0
