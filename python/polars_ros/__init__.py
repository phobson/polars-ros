"""Regression on Order Statistics and bootstrap confidence intervals for Polars.

Import this package to register the expression namespaces::

    import polars as pl
    import polars_ros as plr

    df.with_columns(plr.ros.impute("result", "censored").alias("ros"))

The functions live in the ``plr.ros`` and ``plr.bootstrap`` namespaces and are
implemented in Rust; they are ordinary polars expressions, so they compose with
``select``, ``with_columns``, ``group_by(...).agg(...)`` and friends.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import bootstrap, ros
from .bootstrap import ci, fit
from .ros import (
    cohn_numbers,
    detection_limit_index,
    estimate,
    group_rank,
    impute,
    impute_details,
    is_valid,
    plotting_positions,
    substitute,
    zprelim,
)

try:
    # Cargo.toml is the single source of truth; the build backend copies the
    # version into the installed metadata.
    __version__ = version("polars-ros")
except PackageNotFoundError:  # pragma: no cover - only when run from a source tree
    __version__ = "uninstalled"

__all__ = [
    "bootstrap",
    "ci",
    "cohn_numbers",
    "detection_limit_index",
    "estimate",
    "fit",
    "group_rank",
    "impute",
    "impute_details",
    "is_valid",
    "plotting_positions",
    "ros",
    "substitute",
    "zprelim",
]