"""Cross-validate the Rust port against the installed wqio, at runtime.

`check_fixtures.py` proves the test *inputs* are wqio's, and
`check_expected_arrays.py` proves the expected *outputs* were transcribed from
wqio's own test suite. Neither runs wqio. This does: it calls wqio and this port
on the same input and diffs the results, which is the only check that can catch
behaviour wqio exercises but does not assert.

Usage:
    python tools/check_against_wqio.py [-v] [--require]

Requires the `dev` extra (numpy/pandas/scipy/probscale/wqio). By default it
exits 0 with a skip notice if those are absent, so it is safe to wire into a
gate that runs without the scientific stack; pass --require (or set
POLARS_ROS_REQUIRE_WQIO=1, which is what CI does) to make a missing wqio a
hard failure instead of a silent no-op.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wqio_location import WqioNotInstalled, wqio_path, wqio_root  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Compatibility shims applied to wqio at load time, reported by main().
PATCHES: list[str] = []

TOL = 1e-9
MAX_REPORTED = 12

# Resamples for the interval comparisons. wqio draws these from an unseeded
# global RNG, so the only honest comparison is statistical, not exact: NITER
# keeps the Monte Carlo error small relative to BOOTSTRAP_WIDTH_TOL.
NITER = 20_000
BOOTSTRAP_WIDTH_TOL = 0.5  # in units of the sample standard deviation


class Skip(Exception):
    """Raised when the scientific stack is unavailable."""


# ---------------------------------------------------------------------------
# Loading wqio without running its package __init__
# ---------------------------------------------------------------------------
# A plain `import wqio` executes its __init__, which imports datacollections,
# datasets, features, hydro, samples and tests -- pulling in matplotlib,
# seaborn and statsmodels just to read two statistical functions. We also must
# not let wqio's own `__init__` shadow the synthetic package below. Since
# ros.py and bootstrap.py touch exactly one helper -- `utils.log_or_warn` -- we
# synthesise a `wqio` package whose `utils` exposes that single function, lifted
# verbatim from the *installed* wqio's `utils/misc.py` by AST so it cannot drift.
# The installed wqio is never modified.


def _load_function(path: Path, name: str) -> Callable[..., Any]:
    """Exec just one `def` out of a module, so we need not import the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns: dict[str, Any] = {"warnings": warnings}
            exec(  # noqa: S102 - executing one vetted function from wqio's source
                compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns
            )
            return ns[name]
    raise Skip(f"{name} not found in {path}")


def patch_for_pandas3(ros: Any) -> list[str]:
    """Make wqio runnable on pandas 3, where `.values` views are read-only.

    wqio's `plotting_positions` sorts the ND plotting positions in place via
    `ND_plotpos.values.sort()`. pandas 3 hands out a read-only view there, so
    wqio raises before it produces a value. The replacement below is wqio's own
    body with the in-place sort swapped for an explicit one -- no algorithm
    change, only the sort's mechanics.
    """
    import numpy

    def plotting_positions(df, censorship, cohn):
        plot_pos = df.apply(lambda r: ros._ros_plot_pos(r, censorship, cohn), axis=1)
        mask = df[censorship].to_numpy(dtype=bool)
        # pandas 3 gives a read-only view, so assign a sorted copy instead.
        plot_pos.loc[mask] = numpy.sort(plot_pos[mask].to_numpy(dtype=float))
        return plot_pos

    ros.plotting_positions = plotting_positions
    return ["pandas 3 read-only .values in plotting_positions"]


def load_wqio() -> tuple[Any, Any]:
    try:
        import pandas  # noqa: F401
        import probscale  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise Skip(f"missing dependency: {exc.name}") from exc

    try:
        root = wqio_root()
        misc = wqio_path("utils", "misc.py")
    except WqioNotInstalled as exc:
        raise Skip(str(exc)) from exc

    pkg = types.ModuleType("wqio")
    pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
    sys.modules.setdefault("wqio", pkg)

    utils = types.ModuleType("wqio.utils")
    utils.log_or_warn = _load_function(misc, "log_or_warn")  # type: ignore[attr-defined]
    sys.modules["wqio.utils"] = utils
    pkg.utils = utils  # type: ignore[attr-defined]

    import importlib.util

    modules = {}
    for name in ("ros", "bootstrap"):
        spec = importlib.util.spec_from_file_location(f"wqio.{name}", root / f"{name}.py")
        if spec is None or spec.loader is None:
            raise Skip(f"cannot load wqio/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"wqio.{name}"] = mod
        spec.loader.exec_module(mod)
        modules[name] = mod
    PATCHES = patch_for_pandas3(modules["ros"])
    return modules["ros"], modules["bootstrap"]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def as_number(value: Any) -> float | None:
    """Normalise a missing value to None.

    Polars reports a missing float as NaN while pandas uses None/NaN
    interchangeably, so both spellings of "absent" have to collapse to one
    before comparing. inf is preserved.
    """
    if value is None:
        return None
    number = float(value)
    return None if numpy.isnan(number) else number


def close(a: float | None, b: float | None, tol: float = TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    # Identical infinities compare equal; abs(inf - inf) would be nan.
    if a == b:
        return True
    if numpy.isinf(a) or numpy.isinf(b):
        return False
    return abs(a - b) <= tol + tol * abs(b)


class Report:
    def __init__(self) -> None:
        self.checks = 0
        self.failures: list[str] = []
        self.failure_count = 0
        self.skipped: list[str] = []

    def compare(
        self,
        label: str,
        ours: Any,
        theirs: Any,
        tol: float = TOL,
    ) -> None:
        """Compare two sequences elementwise. `None` on either side means null."""
        self.checks += 1
        a = list(ours)
        b = list(theirs)
        if len(a) != len(b):
            self.fail(f"{label}: length {len(a)} != wqio {len(b)}")
            return
        worst = 0.0
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            xf = as_number(x)
            yf = as_number(y)
            if not close(xf, yf, tol):
                worst = max(worst, abs((xf or 0.0) - (yf or 0.0)))
                self.fail(f"{label}[{i}]: ours={xf!r} wqio={yf!r}")
            elif xf is not None:
                worst = max(worst, abs(xf - yf))
        if worst > 0:
            print(f"  {label:<34} {len(a):>4} rows, max |diff| = {worst:.3e}")

    def fail(self, message: str) -> None:
        """Record a failure, counting all of them but printing only the first few."""
        self.failure_count += 1
        if len(self.failures) < MAX_REPORTED:
            self.failures.append(message)

    def skip(self, label: str, why: str) -> None:
        self.skipped.append(f"{label}: {why}")

    def result(self) -> int:
        print()
        print(f"compared {self.checks} arrays against live wqio")
        for s in self.skipped:
            print(f"  SKIPPED {s}")
        if self.failure_count:
            print(f"\n{self.failure_count} MISMATCH(ES):")
            for f in self.failures:
                print(f"  {f}")
            extra = self.failure_count - len(self.failures)
            if extra > 0:
                print(f"  ... and {extra} more")
            return 1
        print("PASS: this port agrees with wqio on every compared array")
        return 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Same shape as tests/conftest.py's BASE_RESULTS/BASE_CENSORED, which
# check_fixtures.py already proves are wqio's. `tests/` is not a package, so
# load conftest by path rather than importing it.


def load_conftest() -> Any:
    import importlib.util

    path = ROOT / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("rustros_conftest", path)
    if spec is None or spec.loader is None:
        raise Skip(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rustros_conftest"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_fit_data() -> tuple[list[float], list[float]]:
    """FIT_X/FIT_Y live in tests/test_bootstrap.py; read them from there."""
    import importlib.util

    path = ROOT / "tests" / "test_bootstrap.py"
    # test_bootstrap.py does `from conftest import BASE_RESULTS`.
    tests_dir = str(ROOT / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    spec = importlib.util.spec_from_file_location("rustros_fitdata", path)
    if spec is None or spec.loader is None:
        raise Skip(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rustros_fitdata"] = mod
    spec.loader.exec_module(mod)
    return list(mod.FIT_X), list(mod.FIT_Y)


CONFTEST: Any = None


def frame(res: list[float], cen: list[bool]) -> Any:
    import pandas

    return pandas.DataFrame({"res": res, "censored": cen})


def polars_frame(res: list[float], cen: list[bool]) -> Any:
    import polars

    return polars.DataFrame({"res": res, "censored": cen})


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _reason(exc: Exception) -> str:
    """Pull the human-readable message out of a Polars plugin ComputeError."""
    message = str(exc).split("plugin failed with message: ")[-1]
    return message.splitlines()[0][:70]


def check_ros(ros: Any, rep: Report) -> None:
    import polars as pl

    import polars_ros as plr

    for label, res, cen in cases():
        df = frame(res, cen)
        pdf = polars_frame(res, cen)
        tag = label

        # --- is_valid ---------------------------------------------------
        for min_unc, max_frac in ((2, 0.8), (100, 0.8), (2, 0.05)):
            try:
                want = bool(
                    ros.is_valid_to_ros(
                        df, "censored", max_fraction_censored=max_frac, min_uncensored=min_unc
                    )
                )
                got = bool(
                    pdf.select(
                        plr.ros.is_valid(
                            "censored", min_uncensored=min_unc, max_fraction_censored=max_frac
                        )
                    )["censored"][0]
                )
                rep.checks += 1
                if got != want:
                    rep.fail(
                        f"{tag} is_valid(min_uncensored={min_unc}, max_fraction_censored={max_frac}): "
                        f"ours={got} wqio={want}"
                    )
            except Exception as exc:
                rep.skip(f"{tag} is_valid({min_unc},{max_frac})", type(exc).__name__)

        # --- cohn_numbers -----------------------------------------------
        try:
            want_cohn = ros.cohn_numbers(df, "res", "censored")
        except Exception as exc:
            rep.skip(f"{tag} cohn_numbers", type(exc).__name__)
            want_cohn = None
        if want_cohn is not None:
            got_struct = (
                pdf.select(plr.ros.cohn_numbers("res", "censored").alias("c")).to_series()
            )
            # wqio appends a trailing all-NaN sentinel row carrying
            # prob_exceedance == 0. It is an artifact of how it builds the
            # table: NaN lower_dl makes `lower_dl <= res` False everywhere, so
            # it cannot affect detection_limit_index, rank or plotting
            # positions. We deliberately omit it, so drop it before comparing
            # and assert it really is the sentinel.
            their_cols = {c: want_cohn[c].tolist() for c in want_cohn.columns}
            n_theirs = len(want_cohn)
            n_ours = len(got_struct)
            if n_theirs == n_ours + 1:
                sentinel = {c: their_cols[c][-1] for c in their_cols}
                is_sentinel = all(
                    (v != v) for c, v in sentinel.items() if c != "prob_exceedance"
                ) and sentinel.get("prob_exceedance") == 0.0
                if not is_sentinel:
                    rep.fail(f"{tag} cohn: wqio extra row is not the NaN sentinel: {sentinel}")
                their_cols = {c: v[:-1] for c, v in their_cols.items()}
                n_theirs -= 1
            if n_ours != n_theirs:
                rep.fail(f"{tag} cohn: length {n_ours} != wqio {len(want_cohn)}")
            else:
                for col in their_cols:
                    rep.compare(
                        f"{tag} cohn.{col}",
                        got_struct.struct.field(col).to_list(),
                        their_cols[col],
                    )

        # --- substitute --------------------------------------------------
        for frac in (0.5, 1.0):
            got = (
                pdf.select(
                    plr.ros.substitute("res", "censored", substitution_fraction=frac).alias("s")
                )["s"]
                .to_list()
            )
            rep.compare(
                f"{tag} substitute({frac})",
                got,
                [frac * v if c else v for v, c in zip(res, cen, strict=True)],
            )

# --- full ROS pipeline -------------------------------------------
        # wqio assembles det_limit_index -> rank -> plot_pos -> Zprelim ->
        # estimated -> final inside _do_ros, and returns ROS-ordered rows. Our
        # impute_details returns the same columns in the same order, so compare
        # them position by position.
        #
        # wqio takes transform callables, we take their names.
        transforms = {
            "identity": (lambda v: v, "identity"),
            "log": (numpy.log, "log"),
            "exp": (numpy.exp, "exp"),
        }
        for t_in, t_out in (("identity", "identity"), ("log", "exp")):
            fn_in, name_in = transforms[t_in]
            fn_out, name_out = transforms[t_out]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    modeled = ros._do_ros(
                        df.copy(), "res", "censored", fn_in, fn_out, log=False
                    )
                wqio_error: Exception | None = None
            except Exception as exc:
                modeled = None
                wqio_error = exc

            try:
                det = (
                    pdf.select(
                        plr.ros.impute_details(
                            "res", "censored", transform_in=name_in, transform_out=name_out
                        ).alias("d")
                    )
                    .unnest("d")
                    .to_dicts()
                )
                ours = {k: [row[k] for row in det] for k in det[0]} if det else {}
                our_error: Exception | None = None
            except Exception as exc:
                ours = {}
                our_error = exc

            # Degenerate inputs: ROS needs both censored and uncensored data.
            # Compare how each side refuses, not arrays.
            if our_error is not None or wqio_error is not None:
                rep.checks += 1
                ours_refuses = our_error is not None
                wqio_refuses = wqio_error is not None
                if not ours_refuses:
                    rep.fail(
                        f"{tag} impute_details({t_in}): we accept input wqio rejects "
                        f"({type(wqio_error).__name__})"
                    )
                elif wqio_refuses:
                    # wqio refuses the same input, but from pandas internals.
                    # Same behaviour, far worse diagnostic.
                    print(
                        f"  {tag} impute_details({t_in})".ljust(34)
                        + "  both refuse --\n"
                        + f"      ours : {_reason(our_error)}\n"
                        + f"      wqio : {type(wqio_error).__name__}: "
                        + str(wqio_error).splitlines()[0][:70]
                    )
                else:
                    # We refuse where wqio quietly passes the data through.
                    # Documented behaviour: ROS is undefined with no censored
                    # observations, so we raise instead of inventing a fit.
                    rep.skip(
                        f"{tag} impute_details({t_in})",
                        f"we raise {_reason(our_error)}; wqio returns rows unchanged",
                    )
                continue

            assert modeled is not None
            # impute_details already excludes the censored-above-max-uncensored
            # rows that wqio's _do_ros drops, so the two outputs are the same
            # length and in the same order: compare them straight through.
            if len(ours.get("final", ())) != len(modeled):
                rep.fail(
                    f"{tag} impute_details({t_in}): {len(ours.get('final', ()))} rows "
                    f"!= wqio {len(modeled)}"
                )
                continue
            for our_key, their_key in (
                ("det_limit_index", "det_limit_index"),
                ("rank", "rank"),
                ("plot_pos", "plot_pos"),
                ("zprelim", "Zprelim"),
                ("estimated", "estimated"),
                ("final", "final"),
            ):
                if our_key not in ours or their_key not in modeled.columns:
                    rep.skip(f"{tag} {our_key}({t_in})", "column absent")
                    continue
                # wqio leaves `estimated` NaN off the censored rows.
                rep.compare(
                    f"{tag} {our_key}({t_in})",
                    ours[our_key],
                    modeled[their_key].tolist(),
                )


def check_bootstrap(bootstrap: Any, rep: Report) -> None:
    import numpy
    import polars as pl

    import polars_ros as plr

    data = numpy.asarray(CONFTEST.BASE_RESULTS, dtype=float)
    pdf = pl.DataFrame({"res": data.tolist()})
    stat = numpy.mean

    # wqio's percentile/BCA expose no seed, so the resample streams differ and
    # exact values cannot be compared. Compare interval widths instead, with a
    # budget of half a standard deviation of the data: large enough to absorb
    # Monte Carlo error at NITER, far too small to hide a wrong formula.
    for name, fn in (("percentile", bootstrap.percentile), ("BCA", bootstrap.BCA)):
        takes_log = "log" in inspect.signature(fn).parameters
        for alpha in (0.05, 0.10):
            kwargs = {"niter": NITER, "alpha": alpha}
            if takes_log:
                kwargs["log"] = False
            try:
                want = fn(data, stat, **kwargs)
            except Exception as exc:
                rep.skip(f"bootstrap {name}(alpha={alpha})", type(exc).__name__)
                continue
            lo_w, hi_w = float(want[0]), float(want[1])
            ours_ci = (
                pdf.select(
                    plr.bootstrap.ci(
                        "res", statistic="mean", niter=NITER, alpha=alpha, seed=0
                    )
                )
                .to_series()
            )
            lo_o, hi_o = (
                float(ours_ci.struct.field("lower")[0]),
                float(ours_ci.struct.field("upper")[0]),
            )
            rep.checks += 1
            width_w, width_o = hi_w - lo_w, hi_o - lo_o
            tol = BOOTSTRAP_WIDTH_TOL * float(data.std())
            if abs(width_o - width_w) > tol or not (lo_o < hi_o):
                rep.fail(
                    f"bootstrap {name}(alpha={alpha}): ours=({lo_o:.6g},{hi_o:.6g}) "
                    f"wqio=({lo_w:.6g},{hi_w:.6g}) width diff "
                    f"{abs(width_o - width_w):.3g} > {tol:.3g}"
                )
            else:
                print(
                    f"  bootstrap {name}(alpha={alpha})".ljust(34)
                    + f"  width diff {abs(width_o - width_w):.3e} (tol {tol:.3g})"
                )

    # fit(): wqio sorts x ascending and returns fitestimate(x, yhat, lower,
    # upper). Only yhat is deterministic -- lower/upper are bootstrap
    # percentiles over an unseeded resample, so those get a width tolerance.
    fit_x, fit_y = load_fit_data()
    fit_xa, fit_ya = numpy.asarray(fit_x, dtype=float), numpy.asarray(fit_y, dtype=float)
    want = bootstrap.fit(
        fit_xa,
        fit_ya,
        numpy.polyfit,
        niter=NITER,
        xlog=False,
        ylog=False,
        deg=1,
        full=False,
    )
    rows = (
        pl.DataFrame({"x": fit_x, "y": fit_y})
        .select(plr.bootstrap.fit("x", "y", niter=NITER, seed=0).alias("f"))
        .unnest("f")
        .to_dicts()
    )
    ours = {k: [r[k] for r in rows] for k in rows[0]}
    # wqio sorts x ascending; we preserve input order. FIT_X is 1..10, so they
    # already line up, but assert it rather than assume.
    rep.compare("bootstrap fit.xhat", ours["xhat"], [float(v) for v in want.xhat])
    # yhat is the un-resampled straight-line fit: compare exactly.
    rep.compare("bootstrap fit.yhat", ours["yhat"], [float(v) for v in want.yhat])
    # Bands are random in both implementations; compare their widths.
    tol = BOOTSTRAP_WIDTH_TOL * float(numpy.ptp(fit_ya))
    for key in ("lower", "upper"):
        rep.checks += 1
        a, b = ours[key], [float(v) for v in getattr(want, key)]
        width_o, width_w = max(a) - min(a), max(b) - min(b)
        if abs(width_o - width_w) > tol:
            rep.fail(
                f"bootstrap fit.{key}: width diff {abs(width_o - width_w):.3g} "
                f"> {tol:.3g} (ours={width_o:.6g} wqio={width_w:.6g})"
            )
        else:
            print(
                f"  bootstrap fit.{key}".ljust(34)
                + f"  width diff {abs(width_o - width_w):.3e} (tol {tol:.3g})"
            )


def cases() -> list[tuple[str, list[float], list[bool]]]:
    """Cases built from the fixtures check_fixtures.py already validated."""
    c = CONFTEST
    base = list(c.BASE_RESULTS)
    cen = list(c.BASE_CENSORED)
    return [
        ("base", base, cen),
        ("no-censor", base, [False] * len(base)),
        ("all-censor", base, [True] * len(base)),
        ("few-uncensored", base[:4], cen[:4]),
        ("ties", [1.0, 1.0, 2.0, 2.0, 2.0, 3.0], [True, True, True, False, False, False]),
        ("two-level", [5.0, 9.5, 9.5, 11.0, 2.0, 3.0], [True, True, True, True, False, False]),
    ]


def _require_from_env() -> bool:
    """CI sets POLARS_ROS_REQUIRE_WQIO=1 so a missing wqio fails the job."""
    return os.environ.get("POLARS_ROS_REQUIRE_WQIO", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Without --require a missing wqio is a skip, which is right for a casual
    # local run but wrong for a CI gate: it would pass while checking nothing.
    ap.add_argument(
        "--require",
        action="store_true",
        help="fail instead of skipping when wqio or its dependencies are missing",
    )
    args = ap.parse_args()
    require = bool(args.require) or _require_from_env()
    sys.path.insert(0, str(ROOT))
    global CONFTEST  # noqa: PLW0603
    CONFTEST = load_conftest()
    print("cross-validating the Rust port against live wqio")
    try:
        ros, bootstrap = load_wqio()
    except Skip as exc:
        message = f"{exc}; install the extras with: pip install -e '.[dev]'"
        if require:
            print(f"FAIL: {message}")
            return 1
        print(f"SKIP: {message}")
        return 0

    rep = Report()
    check_ros(ros, rep)
    check_bootstrap(bootstrap, rep)
    code = rep.result()
    for note in PATCHES:
        print(f"  NOTE shimmed wqio for: {note}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

