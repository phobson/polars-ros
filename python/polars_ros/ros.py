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

from polars import DataFrame, Expr
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent

__all__ = [
    "estimate",
    "impute",
    "impute_details",
    "substitute",
]

_TRANSFORMS = "identity, log, exp, sqrt, square"


def _cohn_numbers(result: Expr | str, censorship: Expr | str) -> Expr:
    """Calculate Cohn numbers, one row per unique detection limit.

    Parameters
    ----------
    result : polars.Expr or str
        Numeric result values or the name of their column.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column. ``True`` denotes
        a left-censored result.

    Returns
    -------
    polars.Expr
        A struct expression with ``lower_dl``, ``upper_dl``, ``nuncen_above``,
        ``nobs_below``, ``ncen_equal``, and ``prob_exceedance`` fields.

    Notes
    -----
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


def _detection_limit_index(result: Expr | str, cohn: Expr | str) -> Expr:
    """Calculate the index of the applicable detection limit for each result.

    Parameters
    ----------
    result : polars.Expr or str
        Numeric result values or the name of their column.
    cohn : polars.Expr or str
        Cohn-number struct expression, normally from :func:`_cohn_numbers`.

    Returns
    -------
    polars.Expr
        The zero-based detection-limit index for every result.

    Notes
    -----
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


def _group_rank(det_limit_index: Expr | str, censorship: Expr | str) -> Expr:
    """Calculate the observation index within each detection-limit group.

    Parameters
    ----------
    det_limit_index : polars.Expr or str
        Detection-limit indexes, normally from :func:`_detection_limit_index`.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column.

    Returns
    -------
    polars.Expr
        The one-based observation rank within each detection-limit group.

    Notes
    -----
    Mirrors ``wqio.ros.group_rank``; needs ``detection_limit_index`` as its first
    input.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="group_rank",
        args=[det_limit_index, censorship],
    )


def _plotting_positions(
    result: Expr | str,
    censorship: Expr | str,
    cohn: Expr | str | None = None,
) -> Expr:
    """Calculate Hazen-style plotting positions for observed values.

    Parameters
    ----------
    result : polars.Expr or str
        Numeric result values or the name of their column.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column.
    cohn : polars.Expr or str, optional
        Cohn-number struct expression. When omitted, it is calculated from
        ``result`` and ``censorship``.

    Returns
    -------
    polars.Expr
        Plotting positions for the input observations.

    Notes
    -----
    ``cohn`` is optional; when omitted it is computed from ``result`` and
    ``censorship`` for you. Pass it explicitly to avoid recomputing it inside
    the expression.
    """
    if cohn is None:
        cohn = _cohn_numbers(result, censorship)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="plotting_positions",
        args=[result, censorship, cohn],
    )


def _zprelim(plotting_positions: Expr | str) -> Expr:
    """Transform plotting positions to normal scores.

    Parameters
    ----------
    plotting_positions : polars.Expr or str
        Plotting positions or the name of their column.

    Returns
    -------
    polars.Expr
        The normal-score transformation of ``plotting_positions``.
    """
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
    """Estimate censored values with regression on order statistics.

    Parameters
    ----------
    zprelim : polars.Expr or str
        Normal scores, normally from :func:`_zprelim`.
    result : polars.Expr or str
        Numeric result values or the name of their column.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column.
    transform_in : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'log'
        Transformation applied before fitting the regression.
    transform_out : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'exp'
        Transformation applied to fitted values.

    Returns
    -------
    polars.Expr
        Estimated values for censored rows and unchanged values for uncensored
        rows.

    Notes
    -----
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


def _is_valid(
    censorship: Expr | str,
    min_uncensored: int = 2,
    max_fraction_censored: float = 0.8,
) -> Expr:
    """Determine whether ROS is applicable to a group.

    Parameters
    ----------
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column.
    min_uncensored : int, default 2
        Minimum number of uncensored observations required.
    max_fraction_censored : float, default 0.8
        Largest permitted fraction of censored observations.

    Returns
    -------
    polars.Expr
        A scalar Boolean expression indicating whether the group is valid for
        ROS.

    Notes
    -----
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
    """Impute left-censored results with regression on order statistics.

    Parameters
    ----------
    result : polars.Expr or str
        Numeric result values or the name of their column.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column. ``True`` denotes
        a left-censored result.
    min_uncensored : int, default 2
        Minimum number of uncensored observations required for ROS.
    max_fraction_censored : float, default 0.8
        Largest censored fraction allowed for ROS.
    substitution_fraction : float, default 0.5
        Multiplier for the minimum uncensored result when ROS is not
        applicable.
    transform_in : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'log'
        Transformation applied before fitting the regression.
    transform_out : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'exp'
        Transformation applied to fitted values.
    floor : float, optional
        Lower bound for imputed values.

    Returns
    -------
    polars.Expr
        The imputed result expression.

    Notes
    -----
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
    """Replace censored results without fitting a ROS regression.

    Parameters
    ----------
    result : polars.Expr or str
        Numeric result values or the name of their column.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column. ``True`` denotes
        a left-censored result.
    min_uncensored : int, default 2
        Minimum number of uncensored observations. Retained for compatibility
        with :func:`impute`.
    max_fraction_censored : float, default 0.8
        Largest censored fraction. Retained for compatibility with
        :func:`impute`.
    substitution_fraction : float, default 0.5
        Multiplier applied to the minimum uncensored result.
    transform_in : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'log'
        Transformation applied before substitution.
    transform_out : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'exp'
        Transformation applied after substitution.
    floor : float, optional
        Lower bound for substituted values.

    Returns
    -------
    polars.Expr
        The substituted result expression.

    Notes
    -----
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
    """Return ROS intermediate values used by :func:`impute`.

    Parameters
    ----------
    result : polars.Expr or str
        Numeric result values or the name of their column.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column. ``True`` denotes
        a left-censored result.
    min_uncensored : int, default 2
        Minimum number of uncensored observations required for ROS.
    max_fraction_censored : float, default 0.8
        Largest censored fraction allowed for ROS.
    substitution_fraction : float, default 0.5
        Multiplier for the minimum uncensored result when ROS is not
        applicable.
    transform_in : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'log'
        Transformation applied before fitting the regression.
    transform_out : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'exp'
        Transformation applied to fitted values.
    floor : float, optional
        Lower bound for imputed values.

    Returns
    -------
    polars.Expr
        A struct expression containing the ROS intermediate values and final
        result.

    Raises
    ------
    polars.exceptions.ComputeError
        If the input contains no censored values.

    Notes
    -----
    Returns a struct with ``result``, ``censored``, ``det_limit_index``,
    ``rank``, ``plot_pos``, ``zprelim``, ``estimated`` and ``final``. The rows
    are in ROS order -- sorted by result -- matching ``wqio.ros._do_ros``; sort
    afterwards if you need the original order.

    Rows with a missing result are omitted, as are censored results above the
    maximum uncensored result, so the height can be less than the input's.
    If nothing in the input is censored, use ``impute`` instead.
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


def impute_dataframe(
    df: DataFrame,
    result: Expr | str,
    censorship: Expr | str,
    over_columns: Expr | str | list[Expr | str] | None = None,
    min_uncensored: int = 2,
    max_fraction_censored: float = 0.8,
    substitution_fraction: float = 0.5,
    transform_in: str = "log",
    transform_out: str = "exp",
    floor: float | None = None,
    output_name: str = "imputed",
) -> DataFrame:
    """Impute left-censored data in a DataFrame.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing the result and censorship columns.
    result : polars.Expr or str
        Numeric result values or the name of their column.
    censorship : polars.Expr or str
        Boolean censorship flags or the name of their column. ``True`` denotes
        a left-censored result.
    over_columns : polars.Expr or list of polars.Expr or str, optional
        Grouping columns. ROS is applied independently to each group.
    min_uncensored : int, default 2
        Minimum number of uncensored observations required for ROS.
    max_fraction_censored : float, default 0.8
        Largest censored fraction allowed for ROS.
    substitution_fraction : float, default 0.5
        Multiplier for the minimum uncensored result when ROS is not
        applicable.
    transform_in : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'log'
        Transformation applied before fitting the regression.
    transform_out : {'identity', 'log', 'exp', 'sqrt', 'square'}, default 'exp'
        Transformation applied to fitted values.
    floor : float, optional
        Lower bound for imputed values.
    output_name : str, default 'imputed'
        Name for the output column.

    Returns
    -------
    polars.DataFrame
        A DataFrame with the grouping columns, censorship column, and imputed
        result column.
    """

    ros_expr = impute(
        result,
        censorship,
        min_uncensored,
        max_fraction_censored,
        substitution_fraction,
        transform_in,
        transform_out,
        floor,
    )
    if over_columns:
        ros_expr = ros_expr.over(over_columns)

    return df.select(
        over_columns,
        censorship,
        ros_expr.alias(output_name),
    )
