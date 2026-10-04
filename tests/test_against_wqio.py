"""Cross-validate the Rust port against the vendored wqio source.

`tests/test_ros.py` and `tests/test_bootstrap.py` pin numbers transcribed from
wqio's own test suite. That proves we reproduce wqio's *published* answers, but
it cannot catch a case wqio never tests. This module closes that gap by running
the vendored `wqio/ros.py` and `wqio/bootstrap.py` side by side with this port
over a wider set of inputs.

The comparison needs numpy/pandas/scipy/probscale, so it is skipped unless the
`dev` extra is installed:

    pip install -e ".[dev]"

Run it standalone for the full report:

    python tools/check_against_wqio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_against_wqio.py"

pytestmark = pytest.mark.wqio


def _missing_dependency() -> str | None:
    """Return the name of the first missing dev dependency, if any."""
    import importlib.util

    for name in ("numpy", "pandas", "scipy", "probscale"):
        if importlib.util.find_spec(name) is None:
            return name
    return None


@pytest.fixture(scope="module")
def checker():
    """Import tools/check_against_wqio.py, skipping if the extra is absent."""
    missing = _missing_dependency()
    if missing is not None:
        pytest.skip(f"{missing} not installed; run: pip install -e '.[dev]'")
    if not (ROOT / "wqio" / "ros.py").exists():
        pytest.skip("vendored wqio source is not present")

    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_against_wqio
    finally:
        sys.path.remove(str(ROOT / "tools"))
    return check_against_wqio


def test_agrees_with_live_wqio(checker, capsys):
    """Every compared array matches wqio, with the known deviations documented."""
    checker.CONFTEST = checker.load_conftest()
    ros, bootstrap = checker.load_wqio()
    report = checker.Report()
    checker.check_ros(ros, report)
    checker.check_bootstrap(bootstrap, report)

    with capsys.disabled():
        print()
        for skipped in report.skipped:
            print(f"  documented deviation: {skipped}")

    assert report.failure_count == 0, "\n".join(report.failures)
    # Guard against the comparison silently degrading to nothing.
    assert report.checks >= 100, f"only {report.checks} arrays compared"
