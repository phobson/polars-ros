# Changelog

All notable changes to `polars-ros` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-04

First release.

### Added

- `polars_ros.ros`: `cohn_numbers`, `plotting_positions`, `zprelim`,
  `detection_limit_index`, `group_rank`, `substitute`, `estimate`, `impute`,
  `impute_details`, and `is_valid`.
- `polars_ros.bootstrap`: percentile and bias-corrected-accelerated confidence
  intervals, plus `ci` and `fit`.
- Every function is a plain Polars `Expr`, so results compose with `select`,
  `with_columns`, `over`, and `group_by(...).agg(...)`.
- A Rust/PyO3 extension module, distributed as an `abi3-py312` wheel for
    CPython 3.12 and newer.
- Requires Polars 2.0 or newer, but not 3.0. The upper bound exists because the
    extension targets the 0.55 Rust crate line, which is what Python polars 2.0
    links against; a polars built on a different line will not load it. A
    scheduled workflow opens an issue when the two lines diverge.

### Notes

This is a port of the ROS and bootstrapping routines in
[wqio](https://github.com/International-BMP-Database/wqio), by the same author.
wqio is not a runtime dependency; it is used only as a development dependency to
cross-check numerical output against the reference implementation.

Three deliberate departures from wqio, each asserted or reported by
`tools/check_against_wqio.py`:

- Cohn numbers omit wqio's trailing all-`NaN` sentinel row. That row cannot
  affect a detection limit, rank, or plotting position.
- Degenerate input (no censored observations, or no spread in the uncensored
  plotting positions) raises an informative error instead of failing with an
  indexing error deep inside pandas.
- `impute` keeps input row order and represents undefined results as nulls;
  `impute_details` follows wqio and drops censored results above the maximum
  uncensored result.

[Unreleased]: https://github.com/phobson/polars-ros/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/phobson/polars-ros/releases/tag/v0.1.0
