"""Bootstrap tests.

`yhat` is deterministic (it is the fitted line), so it can be compared to
wqio's recorded values exactly. The percentile bounds depend on the random
draw, and this implementation uses ChaCha8 rather than NumPy's generator, so
they are checked statistically instead: same ordering, sensible coverage, and
agreement between repeated seeds.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_ros as plr

from conftest import BASE_RESULTS

# wqio.tests.test_bootstrap
FIT_X = [float(i) for i in range(1, 11)]
FIT_Y = [
    4.527, 3.519, 9.653, 8.036, 10.805, 14.329, 13.508, 11.822, 13.281, 10.410,
]

# wqio's `test_bootstrappers`: 10000 resamples at alpha=0.10, mean statistic.
KNOWN_PERCENTILE_CI = (8.670, 11.647)
KNOWN_BCA_CI = (8.686, 11.661)


def base_frame() -> pl.DataFrame:
    return pl.DataFrame({"res": BASE_RESULTS})


def unpack_ci(df: pl.DataFrame) -> tuple[float, float]:
    """`ci` returns a length-one struct; pull the two numbers out of it.

    A plugin expression takes its output name from its input, so normalise the
    single struct column rather than assuming it is called ``ci``.
    """
    (name,) = df.columns
    row = df.rename({name: "ci"}).unnest("ci")
    assert row.height == 1
    return row["lower"].item(), row["upper"].item()


def test_ci_matches_wqio_percentile() -> None:
    df = base_frame().select(plr.bootstrap.ci("res", niter=10_000, alpha=0.10, seed=0))
    lower, upper = unpack_ci(df)
    # A different RNG draws different resamples, so allow a couple of standard
    # errors of slack rather than wqio's exact numbers.
    assert lower == pytest.approx(KNOWN_PERCENTILE_CI[0], abs=0.15)
    assert upper == pytest.approx(KNOWN_PERCENTILE_CI[1], abs=0.15)


def test_ci_matches_wqio_bca() -> None:
    df = base_frame().select(plr.bootstrap.ci("res", method="bca", niter=10_000, alpha=0.10, seed=0))
    lower, upper = unpack_ci(df)
    assert lower == pytest.approx(KNOWN_BCA_CI[0], abs=0.15)
    assert upper == pytest.approx(KNOWN_BCA_CI[1], abs=0.15)


def test_ci_is_reproducible_with_a_seed() -> None:
    df = base_frame()
    first = df.select(plr.bootstrap.ci("res", niter=2000, seed=7))
    second = df.select(plr.bootstrap.ci("res", niter=2000, seed=7))
    assert unpack_ci(first) == unpack_ci(second)


def test_ci_without_a_seed_still_runs() -> None:
    df = base_frame().select(plr.bootstrap.ci("res", niter=200, seed=None))
    lower, upper = unpack_ci(df)
    assert lower < upper


def test_ci_widens_with_alpha() -> None:
    df = base_frame()
    narrow = unpack_ci(df.select(plr.bootstrap.ci("res", niter=5000, alpha=0.10, seed=1)))
    wide = unpack_ci(df.select(plr.bootstrap.ci("res", niter=5000, alpha=0.01, seed=1)))
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


@pytest.mark.parametrize(
    "statistic",
    ["mean", "median", "sum", "min", "max", "std", "var", "geomean"],
)
def test_ci_supports_every_statistic(statistic: str) -> None:
    df = base_frame().select(plr.bootstrap.ci("res", statistic=statistic, niter=500, seed=0))
    lower, upper = unpack_ci(df)
    assert lower <= upper


def test_min_statistic_ci_is_bounded_by_the_minimum() -> None:
    # A resample only contains the global minimum if it happens to be drawn, so
    # the bounds sit at or just above it rather than exactly on it.
    df = base_frame().select(plr.bootstrap.ci("res", statistic="min", niter=200, seed=0))
    lower, upper = unpack_ci(df)
    assert min(BASE_RESULTS) <= lower <= upper


def test_max_statistic_ci_is_bounded_by_the_maximum() -> None:
    df = base_frame().select(plr.bootstrap.ci("res", statistic="max", niter=200, seed=0))
    lower, upper = unpack_ci(df)
    assert lower <= upper <= max(BASE_RESULTS)


def test_unknown_statistic_raises() -> None:
    with pytest.raises(Exception):
        base_frame().select(plr.bootstrap.ci("res", statistic="mode", niter=10))


def test_unknown_method_raises() -> None:
    with pytest.raises(Exception):
        base_frame().select(plr.bootstrap.ci("res", method="jackknife", niter=10))


def test_ci_per_group() -> None:
    df = pl.DataFrame(
        {
            "site": ["a"] * len(BASE_RESULTS) + ["b"] * len(BASE_RESULTS),
            "res": BASE_RESULTS + [v * 10 for v in BASE_RESULTS],
        }
    )
    result = (
        df.group_by("site", maintain_order=True)
        .agg(plr.bootstrap.ci("res", niter=2000, seed=0).alias("ci"))
        .unnest("ci")
        .sort("site")
    )
    lowers = result["lower"].to_list()
    uppers = result["upper"].to_list()
    # `b` is `a` scaled by ten, so its interval sits well above `a`'s.
    assert lowers[1] > uppers[0]
    assert lowers[0] < uppers[0]


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


@pytest.fixture
def fit_frame() -> pl.DataFrame:
    return pl.DataFrame({"x": FIT_X, "y": FIT_Y})


def unpack_fit(df: pl.DataFrame) -> pl.DataFrame:
    """Pull the four fit columns out of the struct `fit` returns."""
    (name,) = df.columns
    return df.rename({name: "bootstrapped_fit"}).unnest("bootstrapped_fit")


def test_fit_yhat_matches_wqio(fit_frame: pl.DataFrame) -> None:
    expected = [
        5.841, 6.763, 7.685, 8.606, 9.528, 10.450, 11.371, 12.293, 13.215, 14.136,
    ]
    result = unpack_fit(fit_frame.select(plr.bootstrap.fit("x", "y", niter=200)))
    assert result["yhat"].to_list() == pytest.approx(expected, abs=1e-3)


def test_fit_yhat_matches_wqio_in_log_space(fit_frame: pl.DataFrame) -> None:
    expected = [
        4.004, 5.858, 7.318, 8.570, 9.686, 10.706, 11.651, 12.537, 13.374, 14.170,
    ]
    result = unpack_fit(
        fit_frame.select(plr.bootstrap.fit("x", "y", niter=200, xlog=True, ylog=True))
    )
    assert result["yhat"].to_list() == pytest.approx(expected, abs=1e-3)


def test_fit_keeps_input_row_order(fit_frame: pl.DataFrame) -> None:
    result = unpack_fit(fit_frame.select(plr.bootstrap.fit("x", "y", niter=200)))
    assert result["xhat"].to_list() == FIT_X


def test_fit_brackets_the_point_estimate(fit_frame: pl.DataFrame) -> None:
    result = unpack_fit(fit_frame.select(plr.bootstrap.fit("x", "y", niter=2000, seed=0)))
    for yhat, lower, upper in zip(
        result["yhat"], result["lower"], result["upper"], strict=True
    ):
        assert lower <= yhat <= upper


def test_fit_band_narrows_towards_the_centre(fit_frame: pl.DataFrame) -> None:
    # The band is not monotone in x: it narrows away from the left edge, is
    # tightest near the centre of the data, then widens again.
    result = unpack_fit(fit_frame.select(plr.bootstrap.fit("x", "y", niter=2000, seed=0)))
    widths = [
        upper - lower for lower, upper in zip(result["lower"], result["upper"], strict=True)
    ]
    assert all(w > 0 for w in widths)
    assert widths[0] > widths[1]
    assert widths.index(min(widths)) in range(2, len(widths) - 2)


def test_fit_is_reproducible_with_a_seed(fit_frame: pl.DataFrame) -> None:
    first = unpack_fit(fit_frame.select(plr.bootstrap.fit("x", "y", niter=500, seed=3)))
    second = unpack_fit(fit_frame.select(plr.bootstrap.fit("x", "y", niter=500, seed=3)))
    assert first["lower"].to_list() == second["lower"].to_list()
    assert first["upper"].to_list() == second["upper"].to_list()


def test_fit_recovers_a_straight_line() -> None:
    df = pl.DataFrame({"x": FIT_X, "y": [3.0 * x + 1.0 for x in FIT_X]})
    result = unpack_fit(df.select(plr.bootstrap.fit("x", "y", niter=500, seed=0)))
    for want, got in zip([3.0 * x + 1.0 for x in FIT_X], result["yhat"], strict=True):
        assert got == pytest.approx(want, abs=1e-9)
    for lower, upper in zip(result["lower"], result["upper"], strict=True):
        assert lower <= upper


def test_fit_in_log_space_fits_a_power_law() -> None:
    df = pl.DataFrame({"x": FIT_X, "y": [2.0**x for x in FIT_X]})
    result = unpack_fit(df.select(plr.bootstrap.fit("x", "y", niter=500, ylog=True, seed=0)))
    for want, got in zip([2.0**x for x in FIT_X], result["yhat"], strict=True):
        assert got == pytest.approx(want, rel=1e-9)


def test_fit_rejects_a_constant_predictor() -> None:
    df = pl.DataFrame({"x": [1.0] * 5, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(Exception):
        df.select(plr.bootstrap.fit("x", "y", niter=10))


def test_fit_rejects_mismatched_lengths() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    # A length-one literal against a length-three column must be rejected.
    with pytest.raises(Exception, match="lengths don't match"):
        df.select(plr.bootstrap.fit("x", pl.lit(1.0), niter=10))