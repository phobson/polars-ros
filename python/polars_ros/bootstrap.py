"""Bootstrap confidence intervals as polars expressions.

Two things live here:

* ``ci`` -- a bootstrap confidence interval for a statistic of a single column.
* ``fit`` -- a bootstrap confidence band around a straight-line fit.

Both resample with replacement and are therefore stochastic; pass ``seed`` to
make them reproducible. The resampler is ``ChaCha8Rng``, not NumPy's, so
intervals will not agree with ``numpy.random`` to the last digit -- they will
agree on the distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polars import Expr
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent

__all__ = ["ci", "fit"]

_STATISTICS = "mean, median, sum, min, max, std, var, geomean"
_METHODS = "percentile, bca"


def ci(
    values: Expr | str,
    statistic: str = "mean",
    niter: int = 10_000,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: int | None = None,
) -> Expr:
    """A two-sided bootstrap confidence interval for ``statistic(values)``.

    ``method="percentile"`` is the plain percentile interval.
    ``method="bca"`` is bias-corrected and accelerated, falling back to the
    percentile interval when the accelerated interval fails to contain the
    bootstrap mean.

    Returns a length-one struct with ``lower`` and ``upper``. In a ``select`` or
    ``agg`` context that collapses to a single row::

        df.select(plr.bootstrap.ci("conc", statistic="median").alias("ci"))

    For per-group intervals, unnest the struct::

        df.group_by("site").agg(plr.bootstrap.ci("conc", niter=2000).alias("ci")).unnest("ci")

    ``statistic`` must be one of: ``{_STATISTICS}``.
    ``method`` must be one of: ``{_METHODS}``.
    """
    kwargs: dict[str, Any] = {
        "statistic": statistic,
        "method": method,
        "niter": niter,
        "alpha": alpha,
        "seed": seed,
        "xlog": False,
        "ylog": False,
    }
    return register_plugin_function(
        plugin_path=LIB,
        function_name="ci",
        args=[values],
        kwargs=kwargs,
        returns_scalar=True,
    )


def fit(
    x: Expr | str,
    y: Expr | str,
    niter: int = 10_000,
    alpha: float = 0.05,
    xlog: bool = False,
    ylog: bool = False,
    seed: int | None = None,
) -> Expr:
    """A bootstrapped straight-line fit of ``y`` on ``x``, with a confidence band.

    Returns a struct with ``xhat``, ``yhat``, ``lower`` and ``upper``, with one
    row per input row and in the input's row order -- unlike ``wqio.bootstrap.fit``,
    which returns ``x`` sorted ascending.

    ``xlog`` fits against ``log(x)`` and ``ylog`` fits ``log(y)``, back-
    transforming the estimates; set ``ylog=True`` for a power law.

    Add the four columns to a frame with ``with_columns(...).unnest(...)``.
    """
    kwargs: dict[str, Any] = {
        "statistic": "mean",
        "method": "percentile",
        "niter": niter,
        "alpha": alpha,
        "seed": seed,
        "xlog": xlog,
        "ylog": ylog,
    }
    return register_plugin_function(
        plugin_path=LIB,
        function_name="fit",
        args=[x, y],
        kwargs=kwargs,
    )