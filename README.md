# polars_ros

Regression on Order Statistics (ROS) and bootstrap confidence intervals for
[Polars](https://pola.rs), implemented in Rust and exposed as native Polars
expressions.

This is a port of the ROS and bootstrapping code in
[wqio](https://github.com/USEPA/waterquality), aimed at Polars users who do not
want to leave their dataframe pipeline to pandas for the statistics step.

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
- **Row order is preserved.** `wqio` sorts by result before returning; use
  `sort("result")` if you want that.
- Censored results *above* the maximum uncensored result carry no information.
  `wqio` drops those rows entirely; this port keeps them as nulls so the output
  stays aligned with the input.

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

Needs Rust and Python 3.9+. `maturin` drives the build:

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

The Rust side targets Polars 0.55, which is what Python polars 1.44 links
against. The `polars_expr` macro checks the FFI version at load time, so the
Rust `polars` crate must match the installed Python polars.

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
tests/             expected values transcribed from wqio's test suite
tools/             stdlib-only checkers that diff those transcriptions
                    against the vendored wqio sources
```

The Rust core is deliberately free of `DataFrame` knowledge: each file has plain
`Vec`-in/`Vec`-out functions with unit tests, plus a thin layer that adapts them
to Polars `Series`.

## Verifying the transcribed fixtures

The expected values in `tests/` were copied out of `wqio`'s test suite. To prove
nothing drifted during the copy, `tools/` parses `wqio`'s sources with the
standard library `ast` module and diffs them against our literals. These need no
pandas or numpy:

```
python tools/check_fixtures.py         # tests/conftest.py vs wqio/tests/test_ros.py
python tools/check_expected_arrays.py  # every expected array in tests/*.py
```

Two arrays are intentionally not asserted: `NoOp_ZeroND.res` is generated by
`numpy.random.lognormal` and cannot be reproduced bit-for-bit, and the Cohn
numbers for the extra-censored cases are derived rather than transcribed.