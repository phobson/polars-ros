"""ROS tests.

Expected values come from ``wqio/tests/test_ros.py``. The two cases with
censored results above the maximum uncensored result are compared after
dropping nulls, because this implementation keeps those rows (as nulls) instead
of dropping them the way wqio does.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import polars_ros as plr


def test_cohn_numbers_matches_wqio(basic_data: pl.DataFrame) -> None:
    # wqio's `expected_cohn`, minus the trailing all-NaN padding row that only
    # exists so `prob_exceedance[i + 1]` stays in range.
    expected = {
        "lower_dl": [2.0, 5.0, 5.5, 5.75, 9.5, 11.0],
        "upper_dl": [5.0, 5.5, 5.75, 9.5, 11.0, math.inf],
        "nuncen_above": [3.0, 0.0, 2.0, 10.0, 2.0, 11.0],
        "nobs_below": [0.0, 5.0, 6.0, 9.0, 21.0, 24.0],
        "ncen_equal": [0.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        "prob_exceedance": [
            1.0,
            0.77757437070938218,
            0.77757437070938218,
            0.7034324942791762,
            0.37391304347826088,
            0.31428571428571428,
        ],
    }

    result = (
        basic_data.select(plr.ros.cohn_numbers("res", "censored").alias("cohn_numbers"))
        .unnest("cohn_numbers")
    )
    assert result.height == len(expected["lower_dl"])
    for column, values in expected.items():
        got = result[column].to_list()
        for want, have in zip(values, got, strict=True):
            assert have == pytest.approx(want, rel=1e-5, nan_ok=True), column


def test_cohn_numbers_is_empty_without_censoring(basic_data: pl.DataFrame) -> None:
    df = basic_data.with_columns(pl.lit(False).alias("uncensored"))
    result = df.select(plr.ros.cohn_numbers("res", "uncensored"))
    assert result.height == 0


def test_detection_limit_index(ros_sorted_data: pl.DataFrame) -> None:
    # wqio's `intermediate_data` fixture, which is already in ROS order.
    expected = [1, 1, 2, 3, 4, 4, 5]

    # `cohn_numbers` has one row per distinct detection limit, so it is nested
    # rather than materialised as a column.
    result = ros_sorted_data.select(
        plr.ros.detection_limit_index(
            "res", plr.ros.cohn_numbers("res", "censored")
        ).alias("idx"),
    )
    got = result["idx"].to_list()
    # The seven censored values are 5.0, 5.0, 5.5, 5.75, 9.5, 9.5, 11.0 --
    # the same values wqio's `intermediate_data` fixture uses, with the
    # detection-limit indices it records there.
    assert got[:7] == expected


def test_group_rank() -> None:
    # A running count within each (detection limit index, censored) group, which
    # is what `wqio.ros._ros_group_rank` computes with a cumsum.
    df = pl.DataFrame(
        {
            "dl_idx": [1] * 12,
            "censored": [
                False, False, True, False, False, False,
                True, True, True, False, True, False,
            ],
        }
    )
    expected = [1, 2, 1, 3, 4, 5, 2, 3, 4, 6, 5, 7]

    result = df.select(plr.ros.group_rank("dl_idx", "censored").alias("rank"))
    assert result["rank"].to_list() == expected


def test_group_rank_is_per_detection_limit() -> None:
    df = pl.DataFrame(
        {
            "dl_idx": [1, 1, 1, 2, 2, 2],
            "censored": [False] * 6,
        }
    )
    result = df.select(plr.ros.group_rank("dl_idx", "censored").alias("rank"))
    assert result["rank"].to_list() == [1, 2, 3, 1, 2, 3]


def test_plotting_positions(ros_sorted_data: pl.DataFrame) -> None:
    expected = [
        0.07414188, 0.11121281, 0.14828375, 0.14828375, 0.20869565, 0.34285714,
        0.41739130, 0.05560641, 0.11121281, 0.16681922, 0.24713959, 0.27185355,
        0.32652382, 0.35648013, 0.38643645, 0.41639276, 0.44634907, 0.47630539,
        0.50626170, 0.53621802, 0.56617433, 0.59613064, 0.64596273, 0.66583851,
        0.71190476, 0.73809524, 0.76428571, 0.79047619, 0.81666667, 0.84285714,
        0.86904762, 0.89523810, 0.92142857, 0.94761905, 0.97380952,
    ]

    result = ros_sorted_data.select(plr.ros.plotting_positions("res", "censored"))
    got = result["res"].to_list()
    assert len(got) == len(expected)
    for want, have in zip(expected, got, strict=True):
        assert have == pytest.approx(want, abs=1e-7)


def test_zprelim(ros_sorted_data: pl.DataFrame) -> None:
    # The `Zprelim` column of wqio's `advanced_data` fixture.
    expected = [
        -1.44562021, -1.22010353, -1.04382253, -1.04382253, -0.81095536,
        -0.40467790, -0.20857169, -1.59276546, -1.22010353, -0.96681116,
        -0.68351863, -0.60721672, -0.44953240, -0.36788328, -0.28861907,
        -0.21113039, -0.13489088, -0.05942854, 0.015696403, 0.090910169,
        0.166642511, 0.243344267, 0.374443298, 0.428450751, 0.558957865,
        0.637484160, 0.720156617, 0.808074633, 0.902734791, 1.006269985,
        1.121900467, 1.254875912, 1.414746425, 1.622193585, 1.939989611,
    ]

    result = ros_sorted_data.select(
        plr.ros.zprelim(plr.ros.plotting_positions("res", "censored"))
    )
    got = result["res"].to_list()
    for want, have in zip(expected, got, strict=True):
        assert have == pytest.approx(want, abs=1e-7)


def test_impute_matches_wqio(ros_sorted_data: pl.DataFrame) -> None:
    # wqio's `test__do_ros_basic` expected array, in ROS order.
    expected = [
        3.11279729, 3.60634338, 4.04602788, 4.04602788, 4.71008116, 6.14010906,
        6.97841457, 2.00000000, 4.20000000, 4.62000000, 5.57000000, 5.66000000,
        5.86000000, 6.65000000, 6.78000000, 6.79000000, 7.50000000, 7.50000000,
        7.50000000, 8.63000000, 8.71000000, 8.99000000, 9.85000000, 10.82000000,
        11.25000000, 11.25000000, 12.20000000, 14.92000000, 16.77000000,
        17.81000000, 19.16000000, 19.19000000, 19.64000000, 20.18000000, 22.97,
    ]

    result = ros_sorted_data.select(plr.ros.impute("res", "censored").alias("ros"))
    got = result["ros"].to_list()
    for want, have in zip(expected, got, strict=True):
        assert have == pytest.approx(want, abs=1e-7)


def test_impute_floor_matches_wqio(ros_sorted_data: pl.DataFrame) -> None:
    # wqio's `test__do_ros_basic_with_floor`.
    expected = [
        5.0, 5.0, 5.0, 5.0, 5.0, 6.14010906, 6.97841457, 5.0, 5.0, 5.0, 5.57, 5.66,
        5.86, 6.65, 6.78, 6.79, 7.50, 7.50, 7.50, 8.63, 8.71, 8.99, 9.85, 10.82,
        11.25, 11.25, 12.20, 14.92, 16.77, 17.81, 19.16, 19.19, 19.64, 20.18, 22.97,
    ]

    result = ros_sorted_data.select(plr.ros.impute("res", "censored", floor=5.0).alias("ros"))
    got = result["ros"].to_list()
    for want, have in zip(expected, got, strict=True):
        assert have == pytest.approx(want, abs=1e-7)


def test_impute_falls_back_to_substitution() -> None:
    # wqio's `test__do_ros_all_equal_some_cen`: with a constant uncensored pair
    # the log-log regression is degenerate, so substitution takes over.
    df = pl.DataFrame(
        {
            "res": [0.4] * 8,
            "censored": [True, True, True, True, True, False, True, False],
        }
    )
    result = df.select(plr.ros.impute("res", "censored").alias("ros"))
    assert result["ros"].to_list() == pytest.approx([0.4] * 8)


def test_impute_is_a_noop_without_censoring(literature_case) -> None:
    df = literature_case.frame()
    if any(literature_case.cen):
        pytest.skip("case has censored observations")
    result = df.select(plr.ros.impute("res", "censored").alias("ros"))
    assert result["ros"].to_list() == pytest.approx(literature_case.res, abs=1e-9)


def test_impute_from_literature(literature_case) -> None:
    result = (
        literature_case.frame()
        .select(plr.ros.impute("res", "censored").alias("ros"))
        .drop_nulls("ros")
    )
    got = sorted(result["ros"].to_list())
    expected = sorted(literature_case.values)
    assert len(got) == len(expected)
    for want, have in zip(expected, got, strict=True):
        assert have == pytest.approx(want, abs=10 ** -literature_case.decimal)


def test_censored_above_max_uncensored_becomes_null(literature_case) -> None:
    """wqio drops those rows; we keep them, but they carry no information."""
    df = literature_case.frame()
    expected_dropped = max_cen_dropped_count(literature_case)
    if expected_dropped == 0:
        pytest.skip("case has no censored results above the maximum uncensored result")

    result = df.select(plr.ros.impute("res", "censored").alias("ros"))
    assert result.height == df.height
    assert result["ros"].null_count() == expected_dropped


def max_cen_dropped_count(case) -> int:
    if not case.cen:
        return 0
    max_uncensored = max(v for v, c in zip(case.res, case.cen, strict=True) if not c)
    return sum(1 for v, c in zip(case.res, case.cen, strict=True) if c and v > max_uncensored)


def test_cohn_from_literature(literature_case) -> None:
    if not literature_case.cohn:
        pytest.skip("case has no recorded Cohn numbers")

    result = (
        literature_case.frame()
        .select(plr.ros.cohn_numbers("res", "censored").alias("cohn"))
        .unnest("cohn")
    )
    for column, expected in literature_case.cohn.items():
        got = result[column].to_list()
        assert len(got) == len(expected), column
        for want, have in zip(expected, got, strict=True):
            assert have == pytest.approx(want, abs=1e-4), column


def test_is_valid(basic_data: pl.DataFrame) -> None:
    assert basic_data.select(plr.ros.is_valid("censored"))["censored"].item() is True


def test_is_valid_rejects_too_much_censoring(basic_data: pl.DataFrame) -> None:
    got = basic_data.select(plr.ros.is_valid("censored", max_fraction_censored=0.05))
    assert got["censored"].item() is False


def test_is_valid_rejects_too_few_uncensored(basic_data: pl.DataFrame) -> None:
    got = basic_data.select(plr.ros.is_valid("censored", min_uncensored=100))
    assert got["censored"].item() is False


def test_substitute() -> None:
    df = pl.DataFrame({"res": [1.0, 2.0, 4.0], "censored": [True, False, False]})
    result = df.select(plr.ros.substitute("res", "censored", substitution_fraction=0.5))
    # The censored 1.0 is substituted with `substitution_fraction * result`.
    assert result["res"].to_list() == pytest.approx([0.5, 2.0, 4.0])


def test_impute_details_exposes_every_step(ros_sorted_data: pl.DataFrame) -> None:
    result = (
        ros_sorted_data.select(plr.ros.impute_details("res", "censored").alias("detail"))
        .unnest("detail")
    )
    assert result.columns == [
        "result",
        "censored",
        "det_limit_index",
        "rank",
        "plot_pos",
        "zprelim",
        "estimated",
        "final",
    ]
    assert result.height == ros_sorted_data.height
    # `final` is the imputed series, so it must agree with `impute`.
    impute = ros_sorted_data.select(plr.ros.impute("res", "censored").alias("ros"))
    assert result["final"].to_list() == pytest.approx(impute["ros"].to_list(), abs=1e-9)


def test_nulls_propagate() -> None:
    df = pl.DataFrame({"res": [1.0, None, 3.0], "censored": [True, True, False]})
    result = df.select(plr.ros.impute("res", "censored").alias("ros"))
    # Too few uncensored values to ROS-impute, so substitution applies: the
    # censored 1.0 becomes 0.5, and the null stays null.
    assert result["ros"].to_list() == [0.5, None, 3.0]


def test_uncensored_nulls_propagate() -> None:
    # An uncensored null has to be dropped before plotting positions are
    # computed; it used to raise instead of coming back as a null.
    df = pl.DataFrame(
        {"res": [1.0, None, 3.0, 4.0], "censored": [True, False, False, False]}
    )
    got = df.select(plr.ros.impute("res", "censored").alias("ros"))["ros"].to_list()
    assert got[1] is None
    assert got[0] is not None
    assert got[2:] == [3.0, 4.0]


def test_wrong_dtype_raises() -> None:
    df = pl.DataFrame({"res": [1.0, 2.0], "censored": [True, False]})
    with pytest.raises(Exception):
        df.select(plr.ros.impute(pl.col("res").cast(pl.Int32), "censored"))


def test_group_by_keeps_groups_independent() -> None:
    a = pl.DataFrame(
        {"site": ["a"] * 4, "res": [1.0, 2.0, 3.0, 4.0], "censored": [True, True, False, False]}
    )
    b = pl.DataFrame(
        {"site": ["b"] * 4, "res": [10.0, 20.0, 30.0, 40.0], "censored": [True, True, False, False]}
    )
    df = pl.concat([a, b])
    result = (
        df.group_by("site", maintain_order=True)
        .agg(plr.ros.impute("res", "censored").alias("ros"))
        .sort("site")
    )
    assert result["ros"].to_list()[0][2:] == pytest.approx([3.0, 4.0])
    assert result["ros"].to_list()[1][2:] == pytest.approx([30.0, 40.0])