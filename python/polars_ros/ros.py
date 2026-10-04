"""Regression on Order Statistics (ROS) as polars expressions.

Every function here is a thin registration wrapper: it forwards to a Rust
implementation compiled into ``polars_ros._internal`` and returns an ordinary
``polars.Expr``.

The row order is preserved. That is the main departure from ``wqio.ros``, which
sorts by result before returning; use ``sort("result")`` if you need that.

Parameters ``result`` and ``censorship`` are columns. ``censorship`` must be a
Boolean column where ``True`` means "left censored", i.e. the reported value is
a detection limit rather than a measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polars import Expr
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent

__all__ = [
    "cohn_numbers",
    "detection_limit_index",
    "estimate",
    "group_rank",
    "impute",
    "impute_details",
    "is_valid",
    "plotting_positions",
    "substitute",
    "zprelim",
]

_TRANSFORMS = "identity, log, exp, sqrt, square"


def cohn_numbers(result: Expr | str, censorship: Expr | str) -> Expr:
    """Cohn numbers, one row per unique detection limit.

    Returns a struct with ``lower_dl``, ``upper_dl``, ``nuncen_above``,
    ``nobs_below``, ``ncen_equal`` and ``prob_exceedance``.

    Unlike ``wqio.ros.cohn_numbers`` there is no trailing padding row, so the
    struct has one row per real detection limit.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="cohn_numbers",
        args=[result, censorship],
    )


def detection_limit_index(result: Expr | str, cohn: Expr | str) -> Expr:
    """Index of the detection limit that applies to each row of ``result``.

    Mirrors ``wqio.ros._detection_limit_index``: the 0-based position of the
    last Cohn detection limit whose ``lower_dl`` is at or below the value, or
    ``0`` for a value below every detection limit. Censored rows therefore get
    a 1-based index, and wqio's own fixtures are written that way.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="detection_limit_index",
        args=[result, cohn],
    )


def group_rank(det_limit_index: Expr | str, censorship: Expr | str) -> Expr:
    """Observation index within each detection-limit group.

    Mirrors ``wqio.ros.group_rank``; needs ``detection_limit_index`` as its first
    input.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="group_rank",
        args=[det_limit_index, censorship],
    )


def plotting_positions(
    result: Expr | str,
    censorship: Expr | str,
    cohn: Expr | str | None = None,
) -> Expr:
    """Hazen-style plotting positions for the observed values.

    ``cohn`` is optional; when omitted it is computed from ``result`` and
    ``censorship`` for you. Pass it explicitly to avoid recomputing it inside
    the expression.
    """
    if cohn is None:
        cohn = cohn_numbers(result, censorship)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="plotting_positions",
        args=[result, censorship, cohn],
    )


def zprelim(plotting_positions: Expr | str) -> Expr:
    """Normal-score transform of the plotting positions."""
    return register_plugin_function(
        plugin_path=LIB,
        function_name="zprelim",
        args=[plotting_positions],
    )


def estimate(
    zprelim: Expr | str,
    result: Expr | str,
    censorship: Expr | str,
    transform_in: str = "log",
    transform_out: str = "exp",
) -> Expr:
    """ROS-estimated values for the censored rows.

    Fits result on the normal deviates of the uncensored observations, then
    predicts the censored ones. Uncensored rows are passed through unchanged,
    except when ``floor`` is unset and the observation is non-positive -- those
    become null so that ``transform_in="log"`` stays well defined.

    ``transform_in`` and ``transform_out`` must be one of: ``{_TRANSFORMS}``.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="estimate",
        args=[zprelim, result, censorship],
        kwargs={"transform_in": transform_in, "transform_out": transform_out},
    )


def is_valid(
    censorship: Expr | str,
    min_uncensored: int = 2,
    max_fraction_censored: float = 0.8,
) -> Expr:
    """Whether ROS is applicable to this group.

    True when there are at least ``min_uncensored`` uncensored observations and
    no more than ``max_fraction_censored`` of the data is censored.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="is_valid",
        args=[censorship],
        kwargs={
            "min_uncensored": min_uncensored,
            "max_fraction_censored": max_fraction_censored,
        },
        returns_scalar=True,
    )


def impute(
    result: Expr | str,
    censorship: Expr | str,
    min_uncensored: int = 2,
    max_fraction_censored: float = 0.8,
    substitution_fraction: float = 0.5,
    transform_in: str = "log",
    transform_out: str = "exp",
    floor: float | None = None,
) -> Expr:
    """Fill in censored results by regression on order statistics.

    This is the one-shot equivalent of the whole ROS pipeline: Cohn numbers,
    plotting positions, normal scores, regression, substitution. When ROS is not
    applicable the function falls back to simple substitution at
    ``substitution_fraction`` of the minimum uncensored result.

    Rows that ``wqio`` would drop entirely -- censored values above the maximum
    uncensored result, which carry no information -- are kept as nulls here so
    the output stays aligned with the input.

    ``floor`` optionally clamps imputed values from below.
    """
    kwargs: dict[str, Any] = {
        "min_uncensored": min_uncensored,
        "max_fraction_censored": max_fraction_censored,
        "substitution_fraction": substitution_fraction,
        "transform_in": transform_in,
        "transform_out": transform_out,
        "floor": floor,
    }
    return register_plugin_function(
        plugin_path=LIB,
        function_name="impute_expr",
        args=[result, censorship],
        kwargs=kwargs,
    )


def substitute(
    result: Expr | str,
    censorship: Expr | str,
    min_uncensored: int = 2,
    max_fraction_censored: float = 0.8,
    substitution_fraction: float = 0.5,
    transform_in: str = "log",
    transform_out: str = "exp",
    floor: float | None = None,
) -> Expr:
    """Simple substitution, without the ROS regression.

    Censored results become ``substitution_fraction`` times the minimum
    uncensored result. Included in ``impute`` as the fallback path; call it
    directly to compare the two methods.
    """
    kwargs: dict[str, Any] = {
        "min_uncensored": min_uncensored,
        "max_fraction_censored": max_fraction_censored,
        "substitution_fraction": substitution_fraction,
        "transform_in": transform_in,
        "transform_out": transform_out,
        "floor": floor,
    }
    return register_plugin_function(
        plugin_path=LIB,
        function_name="substitute_expr",
        args=[result, censorship],
        kwargs=kwargs,
    )


def impute_details(
    result: Expr | str,
    censorship: Expr | str,
    min_uncensored: int = 2,
    max_fraction_censored: float = 0.8,
    substitution_fraction: float = 0.5,
    transform_in: str = "log",
    transform_out: str = "exp",
    floor: float | None = None,
) -> Expr:
    """Every ROS intermediate value, for inspecting how ``impute`` decided.

    Returns a struct with ``result``, ``censored``, ``det_limit_index``,
    ``rank``, ``plot_pos``, ``zprelim``, ``estimated`` and ``final``. The rows
    are in ROS order -- sorted by result -- matching ``wqio.ros._do_ros``; sort
    afterwards if you need the original order.

    Rows with a missing result are omitted, as are censored results above the
    maximum uncensored result, so the height can be less than the input's.
    Raises if nothing in the input is censored: there would be no regression to
    report, so use ``impute`` for that case.
    """
    kwargs: dict[str, Any] = {
        "min_uncensored": min_uncensored,
        "max_fraction_censored": max_fraction_censored,
        "substitution_fraction": substitution_fraction,
        "transform_in": transform_in,
        "transform_out": transform_out,
        "floor": floor,
    }
    return register_plugin_function(
        plugin_path=LIB,
        function_name="impute_details",
        args=[result, censorship],
        kwargs=kwargs,
    )