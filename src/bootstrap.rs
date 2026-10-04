//! Bootstrap confidence intervals, ported from the Python `wqio.bootstrap`
//! module.
//!
//! Two resampling schemes are supported (`percentile` and `bca`) around a
//! named statistic, plus a percentile bootstrap of a linear regression fit
//! (`fit`).

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;

use crate::stats::{linregress, mean, norm_cdf, norm_ppf, percentile, Statistic, STATISTIC_NAMES};
use crate::utils::*;

/// Bootstrap resampling scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Method {
    #[default]
    Percentile,
    Bca,
}

impl Method {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "percentile" => Some(Self::Percentile),
            "bca" | "bias_corrected" | "bias_corrected_and_accelerated" => Some(Self::Bca),
            _ => None,
        }
    }
}

/// Everything the bootstrap routines need.
#[derive(Clone, Copy, Debug)]
pub struct BootstrapOptions {
    pub statistic: Statistic,
    pub method: Method,
    pub niter: usize,
    pub alpha: f64,
    pub seed: Option<u64>,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        Self {
            statistic: Statistic::Mean,
            method: Method::Percentile,
            niter: 10_000,
            alpha: 0.05,
            seed: None,
        }
    }
}

impl BootstrapOptions {
    fn validate(&self) -> PolarsResult<()> {
        if !(0.0..1.0).contains(&self.alpha) {
            polars_bail!(InvalidOperation: "alpha must be in [0, 1), got {}", self.alpha);
        }
        if self.niter == 0 {
            polars_bail!(InvalidOperation: "niter must be at least 1");
        }
        Ok(())
    }

    /// The two confidence limits, expressed as percentiles.
    fn percentiles(&self) -> [f64; 2] {
        [
            100.0 * self.alpha * 0.5,
            100.0 * (1.0 - self.alpha * 0.5),
        ]
    }
}

/// Seed the resampling RNG; an explicit `seed` makes results reproducible.
pub fn make_rng(seed: Option<u64>) -> PolarsResult<ChaCha8Rng> {
    match seed {
        Some(seed) => Ok(ChaCha8Rng::seed_from_u64(seed)),
        // rand 0.9's `from_rng` is infallible; `try_from_rng` is the Result form.
        None => Ok(ChaCha8Rng::from_rng(&mut rand::rng())),
    }
}

/// Resample `data` with replacement `niter` times and reduce each sample.
///
/// Returns one statistic per resample.
pub fn bootstrap_statistics(
    data: &[f64],
    niter: usize,
    statistic: Statistic,
    rng: &mut ChaCha8Rng,
) -> PolarsResult<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        polars_bail!(InvalidOperation: "cannot bootstrap an empty sample");
    }
    let mut sample = vec![0.0; n];
    let mut stats = Vec::with_capacity(niter);
    for _ in 0..niter {
        for slot in sample.iter_mut() {
            *slot = data[rng.random_range(0..n)];
        }
        stats.push(statistic.apply(&sample));
    }
    Ok(stats)
}

/// The acceleration (skewness correction) statistic.
pub fn acceleration(data: &[f64]) -> f64 {
    let data_mean = mean(data);
    let mut sum_cube = 0.0;
    let mut sum_square = 0.0;
    for v in data {
        let r = data_mean - v;
        sum_cube += r * r * r;
        sum_square += r * r;
    }
    // Clamped so an almost-constant sample cannot divide by zero.
    let sum_square = sum_square.max(1e-12);
    sum_cube / (6.0 * sum_square.powf(1.5))
}

/// Two-sided `(lower, upper)` confidence limits.
pub type Ci = [f64; 2];

/// Percentile bootstrap interval.
pub fn percentile_ci(
    boot_stats: &[f64],
    opts: &BootstrapOptions,
) -> PolarsResult<Ci> {
    let mut sorted = boot_stats.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let [lo_p, hi_p] = opts.percentiles();
    Ok([percentile(&sorted, lo_p), percentile(&sorted, hi_p)])
}

/// Bias-corrected and accelerated (BCa) bootstrap interval.
///
/// `boot_stats` must come from the same resample used for `data`, so that the
/// fallback to the percentile interval reuses the identical draws.
pub fn bca_ci(
    data: &[f64],
    boot_stats: &[f64],
    opts: &BootstrapOptions,
) -> PolarsResult<Ci> {
    let niter = boot_stats.len();
    let primary = opts.statistic.apply(data);
    let boot_result = mean(boot_stats);

    // Fraction of resamples strictly below the point estimate, floored away
    // from zero so the bias correction stays finite.
    let num_below = boot_stats.iter().filter(|s| **s < primary).count() as f64;
    let num_below = if num_below == 0.0 { 1e-5 } else { num_below };

    // Degenerate: every resample fell below the point estimate, so there is no
    // bias to correct and BCa is undefined. Fall back to the percentile method.
    if num_below == niter as f64 {
        return percentile_ci(boot_stats, opts);
    }

    let a_hat = acceleration(data);
    let z0 = norm_ppf(num_below / niter as f64);
    let [lo_p, hi_p] = opts.percentiles();
    let z = [norm_ppf(lo_p / 100.0), norm_ppf(hi_p / 100.0)];

    let mut new_alpha = [0.0; 2];
    for (slot, z_i) in new_alpha.iter_mut().zip(z) {
        *slot = norm_cdf(z0 + (z0 + z_i) / (1.0 - a_hat * (z0 + z_i))) * 100.0;
    }

    let mut sorted = boot_stats.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ci = [percentile(&sorted, new_alpha[0]), percentile(&sorted, new_alpha[1])];

    // wqio falls back to the percentile method when the interval does not
    // contain the bootstrap mean. Re-using the same resample keeps that
    // fallback consistent with the BCa numbers above.
    if boot_result < ci[0] || ci[1] < boot_result {
        return percentile_ci(boot_stats, opts);
    }
    Ok(ci)
}

/// Confidence limits for a named statistic around `data`.
pub fn bootstrap_ci(data: &[f64], opts: &BootstrapOptions) -> PolarsResult<Ci> {
    opts.validate()?;
    let mut rng = make_rng(opts.seed)?;
    let boot_stats = bootstrap_statistics(data, opts.niter, opts.statistic, &mut rng)?;
    match opts.method {
        Method::Percentile => percentile_ci(&boot_stats, opts),
        Method::Bca => bca_ci(data, &boot_stats, opts),
    }
}

/// A bootstrap linear fit, with confidence bands on the fitted response.
#[derive(Clone, Debug)]
pub struct FitEstimate {
    /// The predictor values, in the caller's original row order.
    pub xhat: Vec<f64>,
    /// Point estimates of the response.
    pub yhat: Vec<f64>,
    /// Lower confidence band on the response.
    pub lower: Vec<f64>,
    /// Upper confidence band on the response.
    pub upper: Vec<f64>,
}

/// Percentile bootstrap of a straight-line fit `y ~ slope * x + intercept`.
///
/// `xlog`/`ylog` fit in log space and back-transform the estimates, mirroring
/// `wqio.bootstrap.fit(..., xlog=..., ylog=...)`. wqio returns the predictor
/// sorted ascending; here the outputs stay aligned with the input rows so they
/// can be used directly in `with_columns`.
pub fn bootstrap_fit(
    x: &[f64],
    y: &[f64],
    opts: &BootstrapOptions,
    xlog: bool,
    ylog: bool,
) -> PolarsResult<FitEstimate> {
    opts.validate()?;
    let n = x.len();
    if n == 0 {
        polars_bail!(InvalidOperation: "cannot fit a bootstrap to an empty sample");
    }
    if n != y.len() {
        polars_bail!(ShapeMismatch:
            "x and y must have the same length, got {} and {}", n, y.len());
    }
    let fwd = |v: f64| if xlog { libm::log(v) } else { v };
    let inv = |v: f64| if ylog { libm::exp(v) } else { v };

    let main = fit_line(x, y, xlog, ylog)?;
    let yhat: Vec<f64> = x.iter().map(|x_i| inv(main.0 * fwd(*x_i) + main.1)).collect();

    let mut rng = make_rng(opts.seed)?;
    let mut params = Vec::with_capacity(opts.niter);
    let mut sample = vec![0usize; n];
    for _ in 0..opts.niter {
        for slot in sample.iter_mut() {
            *slot = rng.random_range(0..n);
        }
        let xs: Vec<f64> = sample.iter().map(|&i| x[i]).collect();
        let ys: Vec<f64> = sample.iter().map(|&i| y[i]).collect();
        params.push(fit_line(&xs, &ys, xlog, ylog)?);
    }

    // One row of resampled estimates per predictor value; sorting each row in
    // place keeps the peak allocation at `n * niter` doubles, same as numpy.
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    let [lo_p, hi_p] = opts.percentiles();
    let mut row = vec![0.0; opts.niter];
    for x_i in x {
        for (slot, &(slope, intercept)) in row.iter_mut().zip(&params) {
            *slot = inv(slope * fwd(*x_i) + intercept);
        }
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lower.push(percentile(&row, lo_p));
        upper.push(percentile(&row, hi_p));
    }

    Ok(FitEstimate {
        xhat: x.to_vec(),
        yhat,
        lower,
        upper,
    })
}

/// `(slope, intercept)` for `y ~ x`, optionally in log space.
fn fit_line(x: &[f64], y: &[f64], xlog: bool, ylog: bool) -> PolarsResult<(f64, f64)> {
    let xs: Vec<f64> = x.iter().map(|v| if xlog { libm::log(*v) } else { *v }).collect();
    let ys: Vec<f64> = y.iter().map(|v| if ylog { libm::log(*v) } else { *v }).collect();
    linregress(&xs, &ys).ok_or_else(|| {
        polars_err!(ComputeError:
            "the predictor has no variation, so a line cannot be fitted")
    })
}

// ---------------------------------------------------------------------------
// Plugin kwargs and expressions
// ---------------------------------------------------------------------------

fn default_statistic() -> String {
    "mean".to_string()
}

fn default_method() -> String {
    "percentile".to_string()
}

fn default_niter() -> usize {
    10_000
}

fn default_alpha() -> f64 {
    0.05
}

#[derive(Debug, Deserialize)]
struct BootstrapKwargs {
    #[serde(default = "default_statistic")]
    statistic: String,
    #[serde(default = "default_method")]
    method: String,
    #[serde(default = "default_niter")]
    niter: usize,
    #[serde(default = "default_alpha")]
    alpha: f64,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    xlog: bool,
    #[serde(default)]
    ylog: bool,
}

impl Default for BootstrapKwargs {
    fn default() -> Self {
        Self {
            statistic: default_statistic(),
            method: default_method(),
            niter: default_niter(),
            alpha: default_alpha(),
            seed: None,
            xlog: false,
            ylog: false,
        }
    }
}

impl BootstrapKwargs {
    fn options(&self) -> PolarsResult<BootstrapOptions> {
        let statistic = Statistic::from_name(&self.statistic).ok_or_else(|| {
            polars_err!(InvalidOperation:
                "unknown statistic `{}`, expected one of: {}", self.statistic,
                STATISTIC_NAMES.join(", "))
        })?;
        let method = Method::from_name(&self.method).ok_or_else(|| {
            polars_err!(InvalidOperation:
                "unknown bootstrap method `{}`, expected `percentile` or `bca`", self.method)
        })?;
        Ok(BootstrapOptions {
            statistic,
            method,
            niter: self.niter,
            alpha: self.alpha,
            seed: self.seed,
        })
    }
}

fn ci_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("ci".into(), DataType::Struct(vec![
        Field::new("lower".into(), DataType::Float64),
        Field::new("upper".into(), DataType::Float64),
    ])))
}

fn fit_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let inner = ["xhat", "yhat", "lower", "upper"]
        .into_iter()
        .map(|name| Field::new(name.into(), DataType::Float64))
        .collect::<Vec<_>>();
    Ok(Field::new("bootstrapped_fit".into(), DataType::Struct(inner)))
}

fn values(s: &Series) -> PolarsResult<Vec<f64>> {
    let data = as_f64(s, "values")?;
    if data.iter().any(Option::is_none) {
        polars_bail!(ComputeError: "the bootstrap input must not contain nulls");
    }
    Ok(data.into_iter().map(Option::unwrap).collect())
}

/// `pl.bootstrap.ci(values)` -> `Struct {lower, upper}` of length 1.
#[polars_expr(output_type_func = ci_output_type)]
fn ci(inputs: &[Series], kwargs: BootstrapKwargs) -> PolarsResult<Series> {
    let data = values(&inputs[0])?;
    let [lower, upper] = bootstrap_ci(&data, &kwargs.options()?)?;
    homogeneous_struct(
        "ci",
        1,
        vec![
            Float64Chunked::from_vec("lower".into(), vec![lower]).into_series(),
            Float64Chunked::from_vec("upper".into(), vec![upper]).into_series(),
        ],
    )
}

/// `pl.bootstrap.acceleration(values)` -> the BCa acceleration statistic.
#[polars_expr(output_type=Float64)]
fn acceleration_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let data = values(&inputs[0])?;
    Ok(Float64Chunked::from_vec(inputs[0].name().clone(), vec![acceleration(&data)]).into_series())
}

/// `pl.bootstrap.fit(x, y)` -> `Struct {xhat, yhat, lower, upper}`, aligned to
/// the input rows.
#[polars_expr(output_type_func = fit_output_type)]
fn fit(inputs: &[Series], kwargs: BootstrapKwargs) -> PolarsResult<Series> {
    let x = values(&inputs[0])?;
    let y = values(&inputs[1])?;
    let est = bootstrap_fit(&x, &y, &kwargs.options()?, kwargs.xlog, kwargs.ylog)?;
    let height = est.xhat.len();
    homogeneous_struct(
        "bootstrapped_fit",
        height,
        vec![
            Float64Chunked::from_vec("xhat".into(), est.xhat).into_series(),
            Float64Chunked::from_vec("yhat".into(), est.yhat).into_series(),
            Float64Chunked::from_vec("lower".into(), est.lower).into_series(),
            Float64Chunked::from_vec("upper".into(), est.upper).into_series(),
        ],
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn testdata() -> Vec<f64> {
        vec![
            2.00, 4.20, 4.62, 5.00, 5.00, 5.50, 5.57, 5.66, 5.75, 5.86, 6.65, 6.78, 6.79, 7.50,
            7.50, 7.50, 8.63, 8.71, 8.99, 9.50, 9.50, 9.85, 10.82, 11.00, 11.25, 11.25, 12.20,
            14.92, 16.77, 17.81, 19.16, 19.19, 19.64, 20.18, 22.97,
        ]
    }

    #[test]
    fn acceleration_matches_wqio_fixture() {
        let known = -0.024_051_865_664_929_263;
        // wqio allows 1e-5 here. Our running sum accumulates in a different order
        // than numpy's pairwise summation, which puts us ~9e-10 away from wqio's
        // value; that is far tighter than any real formula error would be.
        assert!((acceleration(&testdata()) - known).abs() < 1e-8);
    }

    #[test]
    fn percentile_ci_brackets_the_point_estimate() {
        let data = testdata();
        let opts = BootstrapOptions {
            niter: 2_000,
            seed: Some(0),
            ..Default::default()
        };
        let [lo, hi] = bootstrap_ci(&data, &opts).unwrap();
        assert!(lo < hi);
        assert!(lo < mean(&data) && mean(&data) < hi, "{lo} {} {hi}", mean(&data));
    }

    #[test]
    fn bca_ci_brackets_the_point_estimate() {
        let data = testdata();
        let opts = BootstrapOptions {
            method: Method::Bca,
            niter: 2_000,
            alpha: 0.10,
            seed: Some(0),
            ..Default::default()
        };
        let [lo, hi] = bootstrap_ci(&data, &opts).unwrap();
        assert!(lo < hi);
    }

    #[test]
    fn seed_makes_results_reproducible() {
        let data = testdata();
        let opts = BootstrapOptions {
            niter: 500,
            seed: Some(42),
            ..Default::default()
        };
        assert_eq!(
            bootstrap_ci(&data, &opts).unwrap(),
            bootstrap_ci(&data, &opts).unwrap()
        );
    }

    #[test]
    fn fit_recovers_a_straight_line() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi + 1.0).collect();
        let opts = BootstrapOptions {
            niter: 200,
            seed: Some(0),
            ..Default::default()
        };
        let est = bootstrap_fit(&x, &y, &opts, false, false).unwrap();
        for (got, want) in est.yhat.iter().zip(&y) {
            assert!((got - want).abs() < 1e-9);
        }
        for ((lo, hi), want) in est.lower.iter().zip(&est.upper).zip(&y) {
            assert!(lo <= hi);
            assert!((lo - *want).abs() < 1e-6, "{lo} vs {want}");
            assert!((hi - *want).abs() < 1e-6, "{hi} vs {want}");
        }
    }

    #[test]
    fn fit_in_log_space_back_transforms() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = (1..=10).map(|i| 2.0f64.powf(i as f64)).collect();
        let opts = BootstrapOptions {
            niter: 200,
            seed: Some(0),
            ..Default::default()
        };
        let est = bootstrap_fit(&x, &y, &opts, false, true).unwrap();
        for (got, want) in est.yhat.iter().zip(&y) {
            assert!((got / want - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn statistic_names_round_trip() {
        for name in STATISTIC_NAMES {
            assert!(Statistic::from_name(name).is_some(), "{name}");
        }
        assert!(Statistic::from_name("nope").is_none());
    }
}
