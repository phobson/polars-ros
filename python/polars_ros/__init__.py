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

__version__ = "0.1.0"

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