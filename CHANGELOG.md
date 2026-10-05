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
- A Rust/PyO3 extension module, distributed as an `abi3-py311` wheel for
    CPython 3.11 and newer.

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
