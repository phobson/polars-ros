# polars_ros

Regression on Order Statistics (ROS) and bootstrap confidence intervals for
[Polars](https://pola.rs), implemented in Rust and exposed as native Polars
expressions.

This is a port of the ROS and bootstrapping code in
[wqio](https://github.com/International-BMP-Database/wqio), aimed at Polars users
who do not want to leave their dataframe pipeline to pandas for the statistics
step.

```python
import polars as pl
import polars_ros as plr

df = pl.read_csv("results.csv")
df = df.with_columns(plr.ros.impute("result", "censored").alias("ros"))
```

Everything is an ordinary `Expr`, so it composes with `select`, `with_columns`,
`over`, and `group_by(...).agg(...)`.

## Why ROS

Environmental monitoring data usually carries a detection limit rather than a
measurement whenever a result is below the instrument's resolution. Those rows
are *left censored*: the reported number is an upper bound, not an observation.
ROS models the uncensored values as normally distributed, fits their rank
against normal deviates, and uses that line to predict where the censored values
would have fallen.

## Conventions

- `censored` is a Boolean column where `True` means "left censored"; the
  reported value is then a detection limit.
- `result` is a `Float64` column. Missing results propagate as nulls.
- **Row order depends on the function.** `impute` returns one row per input row,
  in input order, so it can be used directly in `with_columns`. `impute_details`
  returns the per-observation breakdown in ROS order — every censored row first,
  then the uncensored ones — because that is the order the algorithm works in.
  Use `impute` when you need to stay aligned with the input; use `sort` if you
  want wqio's ordering.
- Censored results *above* the maximum uncensored result carry no information.
  `impute` keeps those rows and sets them to null, so the output stays aligned
  with the input. `impute_details` drops them, exactly as wqio does, so its row
  count can be lower than the input's.

## ROS API

| Function | Returns |
| --- | --- |
| `plr.ros.cohn_numbers(result, censorship)` | struct of Cohn numbers, one row per detection limit |
| `plr.ros.detection_limit_index(result, cohn)` | `UInt32` detection-limit index per row; see below |
| `plr.ros.group_rank(det_limit_index, censorship)` | `UInt32` rank within each detection-limit group |
| `plr.ros.plotting_positions(result, censorship, cohn=None)` | `Float64` plotting positions |
| `plr.ros.zprelim(plotting_positions)` | `Float64` normal deviates |
| `plr.ros.estimate(zprelim, result, censorship, transform_in, transform_out)` | `Float64` ROS estimates |
| `plr.ros.is_valid(censorship, min_uncensored, max_fraction_censored)` | `Boolean`, whether ROS applies |
| `plr.ros.impute(result, censorship, ...)` | `Float64`, the whole pipeline |
| `plr.ros.substitute(result, censorship, ...)` | `Float64`, substitution only |
| `plr.ros.impute_details(result, censorship, ...)` | struct of every intermediate step |

`cohn_numbers` returns one row per *distinct* detection limit, which is usually
fewer than the number of censored observations. `detection_limit_index` maps a
full-length `result` onto that shorter struct, so nest the two calls rather than
materialising `cohn` as a column:

```python
df.select(
    plr.ros.detection_limit_index(
        "result", plr.ros.cohn_numbers("result", "censored")
    ).alias("dl_idx")
)
```

The index follows `wqio.ros._detection_limit_index` exactly, which applies *no*
censoring mask: it is the 0-based position of the last Cohn row whose `lower_dl`
is at or below the value, or `0` for a value below every detection limit. Censored
observations therefore get a 1-based index, but so does any uncensored value that
happens to sit at or above the lowest `lower_dl`.

`impute` is the one you usually want:

```python
df.with_columns(
    plr.ros.impute(
        "result",
        "censored",
        min_uncensored=2,
        max_fraction_censored=0.8,
        substitution_fraction=0.5,
        transform_in="log",
        transform_out="exp",
        floor=None,
    ).alias("ros")
)
```

When there are too few uncensored observations, or too many censored ones, it
falls back to simple substitution at `substitution_fraction` of the minimum
uncensored result. `transform_in` and `transform_out` must be one of
`identity`, `log`, `exp`, `sqrt`, `square`.

Use `impute_details` to see why a row came out the way it did:

```python
df.select(plr.ros.impute_details("result", "censored").alias("d")).unnest("d")
```

`impute_details` returns its rows in ROS order (censored first, then
uncensored, each block ascending), matching `wqio`. Rows with a missing result,
and censored results above the maximum uncensored result, are left out, so the
height can be smaller than the input's. It raises if nothing in the input is
censored, since there would be no regression to report — `impute` is the
function to reach for in that case.

## Bootstrap API

```python
plr.bootstrap.ci(values, statistic="mean", niter=10_000, alpha=0.05,
                 method="percentile", seed=None)
plr.bootstrap.fit(x, y, niter=10_000, alpha=0.05, xlog=False, ylog=False,
                  seed=None)
```

`ci` returns a length-one struct with `lower` and `upper`. In a `select` or
`agg` it collapses to a single row; for per-group intervals unnest it:

```python
df.group_by("site").agg(
    plr.bootstrap.ci("conc", statistic="median", niter=2000).alias("ci")
).unnest("ci")
```

- `statistic` is one of `mean`, `median`, `sum`, `min`, `max`, `std`, `var`,
  `geomean`.
- `method` is `percentile` or `bca` (bias-corrected and accelerated). `bca`
  falls back to the percentile interval when the accelerated interval does not
  contain the bootstrap mean.
- `fit` returns a struct with `xhat`, `yhat`, `lower` and `upper`, one row per
  input row and in the input's order. Set `ylog=True` to fit a power law.

Resampling uses `ChaCha8Rng`, not NumPy's generator, so intervals will not agree
with `numpy.random` to the last digit. Pass `seed` for reproducible runs.

## Building from source

Needs Rust and Python 3.11+. `maturin` drives the build:

```bash
python -m venv .venv
.venv/Scripts/pip install maturin polars
.venv/Scripts/python -m maturin develop      # build in place
.venv/Scripts/python -m pytest               # run the tests
```

Point `maturin` at your virtualenv explicitly. If `VIRTUAL_ENV` is unset it will
happily install into whatever Python is first on `PATH`, and a stray `CONDA_PREFIX`
can send it to the wrong environment entirely:

```bash
export VIRTUAL_ENV="$PWD/.venv"
unset CONDA_PREFIX
```

The Rust side targets the Polars 0.55 crates, which is the line Python polars 2.0
links against. The `polars_expr` macro checks the FFI version at load time, so the
Rust `polars` crate has to stay in the same line as the installed Python polars.
A Python release that moves to a new Rust crate needs a rebuild of this one first.

That constraint is why the runtime dependency is `polars>=2.0,<3`. The upper
bound is deliberate: without it, pip could hand you a polars this wheel refuses
to register against. A scheduled workflow compares the two lines and opens an
issue when they diverge, so the rebuild is prompted rather than discovered.

`sysinfo 0.39.6` requires **rustc 1.95 or newer**. If your default toolchain is
older, pin a newer one with a `rust-toolchain.toml` rather than downgrading the
dependency — `Cargo.lock` is committed deliberately, since this crate builds a
native extension and reproducible builds depend on it.

On this machine `.cargo/config.toml` points at the VS2017-era linker and
Windows SDK 8.1 libraries that ship with Visual Studio 2022 Community, because
no "Desktop development with C++" workload is installed. Delete that file on a
machine with a normal C++ toolchain. `tools/msvc-env.ps1` wraps that setup
along with the toolchain pin; it is machine-specific and is the only file here
you will likely need to rewrite.

## Layout

```
src/stats.rs       normal CDF/PPF (AS241), numpy-compatible percentiles, reducers
src/ros.rs         Cohn numbers through to the final imputed values
src/bootstrap.rs   percentile and BCa intervals, bootstrapped linear fits
src/utils.rs       Series <-> Vec glue and struct construction
python/polars_ros/ the expression wrappers registered with Polars
tests/             expected values transcribed from wqio's test suite,
                    plus a live cross-validation against installed wqio
tools/             stdlib-only checkers that diff those transcriptions
                    against the installed wqio sources, and the live
                    cross-validator (needs the dev extra)
```

The Rust core is deliberately free of `DataFrame` knowledge: each file has plain
`Vec`-in/`Vec`-out functions with unit tests, plus a thin layer that adapts them
to Polars `Series`.

## Verifying the transcribed fixtures

The expected values in `tests/` were copied out of `wqio`'s test suite. To prove
nothing drifted during the copy, `tools/` parses the installed `wqio`'s sources
with the standard library `ast` module and diffs them against our literals. These
need no pandas or numpy:

```
python tools/check_fixtures.py         # tests/conftest.py vs wqio/tests/test_ros.py
python tools/check_expected_arrays.py  # every expected array in tests/*.py
```

wqio is located on disk with `importlib.util.find_spec` rather than imported, so
neither checker runs `wqio/__init__.py` and neither needs the scientific stack.

Two arrays are intentionally not asserted: `NoOp_ZeroND.res` is generated by
`numpy.random.lognormal` and cannot be reproduced bit-for-bit, and the Cohn
numbers for the extra-censored cases are derived rather than transcribed.

## Cross-validating against live wqio

The fixture checkers above prove we match the numbers wqio *publishes*. To also
cover inputs wqio never tests, `tools/check_against_wqio.py` imports the
installed `wqio/ros.py` and `wqio/bootstrap.py` and runs them side by side with
this port, comparing every output array:

```
pip install -e ".[dev]"
python tools/check_against_wqio.py    # full report
pytest tests/test_against_wqio.py     # same comparison as a test
```

It compares Cohn numbers, plotting positions, `zprelim`, detection-limit
indices, ranks, substituted values, the full ROS pipeline, and bootstrap
percentile/BCa intervals and fits, across six datasets: the transcribed fixture,
an uncensored set, an all-censored set, a set with few uncensored values, one
with ties, and a two-level set.

A plain `import wqio` would run its `__init__.py`, pulling in matplotlib,
seaborn and statsmodels just to reach two statistical functions. So the tool
loads `ros.py` and `bootstrap.py` by path behind a synthetic `wqio` package and
supplies the one helper (`utils.log_or_warn`) they actually use.

By default the tool exits 0 with a skip notice when wqio or its dependencies are
missing, which keeps it safe to wire into a gate that runs without the scientific
stack. Pass `--require` (CI sets `POLARS_ROS_REQUIRE_WQIO=1`, which does the same
thing) to make a missing wqio a hard failure instead of a silent no-op.

Because the checkers track whatever wqio is installed, a new wqio release may
change its fixtures or outputs. If the checkers fail after a `pip install -U`,
that is the signal to re-verify and re-transcribe, not a bug in this port.

Three deliberate differences from wqio are asserted or reported rather than
matched:

- **Cohn numbers have no sentinel row.** wqio appends a trailing all-`NaN` row
  carrying `prob_exceedance = 0`. Because its `lower_dl` is `NaN`, it can never
  satisfy `lower_dl <= res`, so it cannot affect the detection-limit index,
  rank, or plotting positions. The tool verifies the extra row really is that
  sentinel, then compares the real rows.
- **Degenerate input raises instead of passing through.** With no censored
  observations ROS is undefined, so this port raises rather than returning an
  invented fit. It refuses the same inputs wqio does, but says why:
  `ROS needs at least one censored observation, but every value is uncensored`,
  where wqio surfaces `IndexError: single positional indexer is out-of-bounds`.
- **Bootstrap bands are compared statistically.** wqio resamples from an
  unseeded global `numpy.random`, so interval *widths* are compared within half
  a sample standard deviation at `niter=20000`. Point estimates that do not
  depend on resampling (`fit.yhat`, `xhat`) are compared exactly.

Two shims are applied to wqio at load time and reported at the end of the run:
`plotting_positions` sorts `ND_plotpos.values.sort()` in place, which raises on
pandas 3 because the view is read-only, so the tool substitutes `numpy.sort`
over the same mask.
