"""Test data.

The expected values are copied from ``wqio/tests/test_ros.py`` and
``wqio/tests/test_bootstrap.py`` rather than computed by calling ``wqio``, so
these tests stay independent of wqio being installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
import pytest

# wqio.tests.helpers.getTestROSData()
BASE_RESULTS = [
    2.00, 4.20, 4.62, 5.00, 5.00, 5.50, 5.57, 5.66, 5.75, 5.86, 6.65, 6.78, 6.79, 7.50,
    7.50, 7.50, 8.63, 8.71, 8.99, 9.50, 9.50, 9.85, 10.82, 11.00, 11.25, 11.25, 12.20,
    14.92, 16.77, 17.81, 19.16, 19.19, 19.64, 20.18, 22.97,
]

# `qual == "ND"` in the same CSV.
BASE_CENSORED = [
    False, False, False, True, True, True, False, False, True, False, False, False,
    False, False, False, False, False, False, False, True, True, False, False, True,
    False, False, False, False, False, False, False, False, False, False, False,
]


@pytest.fixture
def basic_data() -> pl.DataFrame:
    """The `getTestROSData` fixture, in its original (unsorted) row order."""
    return pl.DataFrame({"res": BASE_RESULTS, "censored": BASE_CENSORED})


@pytest.fixture
def ros_sorted_data() -> pl.DataFrame:
    """The same data in wqio's ROS order: censored first, then uncensored.

    Each block ascending. `wqio.ros._ros_sort` produces exactly this order, and
    the expected arrays in wqio's own tests are written in it.
    """
    order = sorted(
        range(len(BASE_RESULTS)),
        key=lambda i: (not BASE_CENSORED[i], BASE_RESULTS[i]),
    )
    return pl.DataFrame(
        {
            "res": [BASE_RESULTS[i] for i in order],
            "censored": [BASE_CENSORED[i] for i in order],
        }
    )


@dataclass
class LiteratureCase:
    """One of the worked examples from Helsel's writing.

    `values` is what `wqio.ros.ROS` produced; entries that this implementation
    reports as null (censored results above the maximum uncensored result, which
    wqio drops outright) are simply absent from it.
    """

    res: list[float]
    cen: list[bool]
    values: list[float]
    decimal: int = 2
    cohn: dict[str, list[float]] = field(default_factory=dict)

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame({"res": self.res, "censored": self.cen})


# Appendix B dataset from "Estimation of Descriptive Statistics for Multiply
# Censored Water Quality Data", Water Resources Research, Vol 24, No 12,
# pp 1997-2004. December 1988.
HELSEL_APPENDIX_B = LiteratureCase(
    res=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0, 20.0,
        27.0, 33.0, 50.0,
    ],
    cen=[True] * 9 + [False] * 9,
    values=[
        0.47, 0.85, 1.11, 1.27, 1.76, 2.34, 2.50, 3.00, 3.03, 4.80, 7.00, 9.00, 12.0,
        15.0, 20.0, 27.0, 33.0, 50.0,
    ],
    cohn={
        "nuncen_above": [3.0, 6.0],
        "nobs_below": [6.0, 12.0],
        "ncen_equal": [6.0, 3.0],
        "prob_exceedance": [0.5555, 0.3333],
    },
)

# Oahu arsenic data from "Nondetects and Data Analysis", Helsel (2005).
HELSEL_ARSENIC = LiteratureCase(
    res=[
        3.2, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.7, 1.5, 1.0, 1.0, 1.0,
        1.0, 0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5,
    ],
    cen=[
        False, False, True, True, True, True, True, True, True, True, False, False, True,
        True, True, True, False, True, False, False, False, False, False, False,
    ],
    values=[
        3.20, 2.80, 1.42, 1.14, 0.95, 0.81, 0.68, 0.57, 0.46, 0.35, 1.70, 1.50, 0.98,
        0.76, 0.58, 0.41, 0.90, 0.61, 0.70, 0.70, 0.60, 0.50, 0.50, 0.50,
    ],
    cohn={
        "nuncen_above": [6.0, 1.0, 2.0, 2.0],
        "nobs_below": [0.0, 7.0, 12.0, 22.0],
        "ncen_equal": [0.0, 1.0, 4.0, 8.0],
        "prob_exceedance": [1.0, 0.3125, 0.2143, 0.0833],
    },
)

# The `RNADAdata` block from wqio's test suite.
RNADA = LiteratureCase(
    res=[
        0.09, 0.09, 0.09, 0.101, 0.136, 0.34, 0.457, 0.514, 0.629, 0.638, 0.774, 0.788,
        0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 2.0, 2.0, 2.404, 2.86, 3.0, 3.0, 3.705, 4.0, 5.0,
        5.96, 6.0, 7.214, 16.0, 17.716, 25.0, 51.0,
    ],
    cen=[
        True, True, True, False, False, False, False, False, False, False, False, False,
        True, True, True, True, True, True, True, True, False, True, True, True, True,
        True, True, True, True, True, True, True, True, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False, False,
        False, False,
    ],
    values=[
        0.01907990, 0.03826254, 0.06080717, 0.10100000, 0.13600000, 0.34000000,
        0.45700000, 0.51400000, 0.62900000, 0.63800000, 0.77400000, 0.78800000,
        0.08745914, 0.25257575, 0.58544205, 0.01711153, 0.03373885, 0.05287083,
        0.07506079, 0.10081573, 1.00000000, 0.13070334, 0.16539309, 0.20569039,
        0.25257575, 0.30725491, 0.37122555, 0.44636843, 0.53507405, 0.64042242,
        0.76644378, 0.91850581, 1.10390531, 1.10000000, 2.00000000, 2.00000000,
        2.40400000, 2.86000000, 3.00000000, 3.00000000, 3.70500000, 4.00000000,
        5.00000000, 5.96000000, 6.00000000, 7.21400000, 16.00000000, 17.71600000,
        25.00000000, 51.00000000,
    ],
    decimal=3,
    cohn={
        "nuncen_above": [9.0, 0.0, 18.0],
        "nobs_below": [3.0, 15.0, 32.0],
        "ncen_equal": [3.0, 3.0, 17.0],
        "prob_exceedance": [0.84, 0.36, 0.36],
    },
)

# Nothing is censored, so ROS is a no-op.
#
# wqio builds `res` from `numpy.random.seed(0); numpy.random.lognormal(size=20)`,
# which is not reproducible without numpy. Its own expected `values` are the
# same draws rounded to two decimals, and nothing is censored, so the pipeline
# is the identity: using the rounded values as input checks the same property.
NO_OP_ZERO_ND = LiteratureCase(
    res=[
        0.38, 0.43, 0.81, 0.86, 0.90, 1.13, 1.15, 1.37, 1.40, 1.49, 1.51, 1.56, 2.14,
        2.59, 2.66, 4.28, 4.46, 5.84, 6.47, 9.4,
    ],
    cen=[False] * 20,
    values=[
        0.38, 0.43, 0.81, 0.86, 0.90, 1.13, 1.15, 1.37, 1.40, 1.49, 1.51, 1.56, 2.14,
        2.59, 2.66, 4.28, 4.46, 5.84, 6.47, 9.4,
    ],
)

ONE_ND = LiteratureCase(
    res=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0,
        20.0, 27.0, 33.0, 50.0,
    ],
    cen=[True] + [False] * 17,
    values=[
        0.24, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0,
        20.0, 27.0, 33.0, 50.0,
    ],
    decimal=3,
    cohn={
        "nuncen_above": [17.0],
        "nobs_below": [1.0],
        "ncen_equal": [1.0],
        "prob_exceedance": [0.9444],
    },
)

# Half the detection limits hold 15 of 18 results, so ROS is not applicable and
# simple substitution takes over.
HALF_DLS_80PCT_NDS = LiteratureCase(
    res=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0,
        20.0, 27.0, 33.0, 50.0,
    ],
    cen=[True] * 15 + [False] * 3,
    values=[
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0, 5.0, 5.0, 1.5, 3.5, 4.5, 6.0, 7.5, 10.0,
        27.0, 33.0, 50.0,
    ],
    decimal=3,
    cohn={
        "nuncen_above": [0.0] * 7 + [3.0],
        "nobs_below": [6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 15.0],
        "ncen_equal": [6.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0],
        "prob_exceedance": [0.1667] * 8,
    },
)

HAFL_DLS_ONE_UNCENSORED = LiteratureCase(
    res=[1.0, 1.0, 12.0, 15.0],
    cen=[True, True, True, False],
    values=[0.5, 0.5, 6.0, 15.0],
    decimal=3,
    cohn={
        "nuncen_above": [0.0, 1.0],
        "nobs_below": [2.0, 3.0],
        "ncen_equal": [2.0, 1.0],
        "prob_exceedance": [0.25, 0.25],
    },
)


def _with_extra_censored(case: LiteratureCase, res, cen) -> LiteratureCase:
    """Extend a case with censored results above the maximum uncensored one.

    wqio drops these rows and its expected arrays are unchanged, which is how
    the two `MaxCen_GT_MaxUncen` / `OnlyDL_GT_MaxUncen` cases behave.
    """
    return LiteratureCase(res=res, cen=cen, values=case.values, decimal=case.decimal)


MAX_CEN_GT_MAX_UNCEN = _with_extra_censored(
    HELSEL_APPENDIX_B,
    res=HELSEL_APPENDIX_B.res + [60.0, 70.0],
    cen=HELSEL_APPENDIX_B.cen + [True, True],
)

ONLY_DL_GT_MAX_UNCEN = _with_extra_censored(
    NO_OP_ZERO_ND,
    res=NO_OP_ZERO_ND.res + [10.0, 10.0],
    cen=NO_OP_ZERO_ND.cen + [True, True],
)

LITERATURE_CASES = {
    "helsel_appendix_b": HELSEL_APPENDIX_B,
    "helsel_arsenic": HELSEL_ARSENIC,
    "rnada": RNADA,
    "no_op_zero_nd": NO_OP_ZERO_ND,
    "one_nd": ONE_ND,
    "half_dls_80pct_nds": HALF_DLS_80PCT_NDS,
    "hafl_dls_one_uncensored": HAFL_DLS_ONE_UNCENSORED,
    "max_cen_gt_max_uncen": MAX_CEN_GT_MAX_UNCEN,
    "only_dl_gt_max_uncen": ONLY_DL_GT_MAX_UNCEN,
}


@pytest.fixture(params=list(LITERATURE_CASES), ids=list(LITERATURE_CASES))
def literature_case(request) -> LiteratureCase:
    return LITERATURE_CASES[request.param]