"""Locate the *installed* wqio package without importing it.

The tools under `tools/` need wqio's sources on disk to parse them with `ast`,
but they deliberately stay stdlib-only. Importing `wqio` would execute its
`__init__.py`, which pulls in matplotlib, seaborn, statsmodels and the whole
`wqio.datasets` tree just to read a couple of literals.

`importlib.util.find_spec` locates the package without executing it, so the
checkers stay dependency-free and fast. It only works if wqio is actually
installed, hence the dev extra.

Requires wqio>=0.7.2,<1.0, the first release that ships `wqio/tests/`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

#: Minimum wqio whose wheel contains `wqio/tests/`, needed by the AST checkers.
MIN_VERSION = (0, 7, 2)


class WqioNotInstalled(ImportError):
    """Raised when wqio is missing or too old for the on-disk checks."""


def wqio_root() -> Path:
    """Return the installed `wqio` package directory."""
    try:
        spec = importlib.util.find_spec("wqio")
    except ValueError as exc:  # partially initialised, or broken install
        raise WqioNotInstalled(f"could not locate wqio: {exc}") from exc
    if spec is None or not spec.submodule_search_locations:
        raise WqioNotInstalled(
            "wqio is not installed; run: pip install -e '.[dev]'"
        )
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise WqioNotInstalled(f"wqio has multiple locations: {locations}")
    return Path(locations[0])


def wqio_path(*parts: str) -> Path:
    """Return a path inside the installed wqio package.

    Verifies the file exists so a renamed or dropped file fails loudly with the
    path that was actually searched, rather than an opaque FileNotFoundError.
    """
    path = wqio_root().joinpath(*parts)
    if not path.exists():
        raise WqioNotInstalled(
            f"expected {path} in the installed wqio; a version older than "
            f"{'.'.join(map(str, MIN_VERSION))} does not ship its test sources"
        )
    return path


def wqio_source(*parts: str) -> str:
    """Return the text of a UTF-8 source file inside the installed wqio."""
    return wqio_path(*parts).read_text(encoding="utf-8")
