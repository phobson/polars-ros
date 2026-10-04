//! Small self-contained numerical helpers.
//!
//! The ROS and bootstrap routines only need a handful of primitives that are
//! not exposed by `polars`/`arrow`: the standard normal CDF and its inverse,
//! numpy-compatible percentiles and a couple of reducers. Everything lives here
//! so the algorithms in [`crate::ros`] and [`crate::bootstrap`] stay readable.

use std::f64::consts::SQRT_2;

/// Cumulative distribution function of the standard normal distribution.
///
/// `0.5 * erfc(-x / sqrt(2))`, which stays accurate in the far left tail where
/// `1 - erf` would round to zero.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

/// Inverse of [`norm_cdf`], i.e. the standard normal quantile function.
///
/// Wichura's algorithm AS241 (PPND16), which is what SciPy uses. Accurate to
/// roughly 1e-16 relative, so it matches `scipy.stats.norm.ppf` to well beyond
/// the precision that ROS plotting positions care about.
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        q * (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0)
            / (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r
                + 1.0)
    } else {
        // Work with the distance from the nearer tail.
        let mut r = if q < 0.0 { p } else { 1.0 - p };
        r = libm::sqrt(-libm::log(r));
        let val = if r <= 5.0 {
            r -= 1.6;
            (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0)
                / (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1)
                    * r
                    + 1.0)
        } else {
            r -= 5.0;
            (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0)
                / ((((((F6 * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + 1.0)
        };
        if q < 0.0 {
            -val
        } else {
            val
        }
    }
}

#[allow(clippy::excessive_precision)]
const A0: f64 = 3.387_132_872_796_366_608;
const A1: f64 = 133.141_667_891_784_377_45;
const A2: f64 = 1_971.590_950_306_551_442_7;
const A3: f64 = 13_731.693_765_509_461_125;
const A4: f64 = 45_921.953_931_549_871_457;
const A5: f64 = 67_265.770_927_008_700_853;
const A6: f64 = 33_430.575_583_588_128_105;
const A7: f64 = 2_509.080_928_730_122_672_7;

const B1: f64 = 42.313_330_701_600_911_252;
const B2: f64 = 687.187_007_492_057_908_3;
const B3: f64 = 5_394.196_021_424_751_107_7;
const B4: f64 = 21_213.794_301_586_595_867;
const B5: f64 = 39_307.895_800_092_710_61;
const B6: f64 = 28_729.085_735_721_942_674;
const B7: f64 = 5_226.495_278_852_854_561;

const C0: f64 = 1.423_437_110_749_683_577_34;
const C1: f64 = 4.630_337_846_156_545_295_9;
const C2: f64 = 5.769_497_221_460_691_405_5;
const C3: f64 = 3.647_848_324_763_204_605_04;
const C4: f64 = 1.270_458_252_452_368_382_58;
const C5: f64 = 0.241_780_725_177_450_611_77;
const C6: f64 = 0.022_723_844_989_269_184_583_3;
const C7: f64 = 7.745_450_142_783_414_076_4e-4;

const D1: f64 = 2.053_191_626_637_758_821_87;
const D2: f64 = 1.676_384_830_183_803_849_4;
const D3: f64 = 0.689_767_334_985_100_004_55;
const D4: f64 = 0.148_103_976_427_480_074_59;
const D5: f64 = 0.015_198_666_563_616_457_196_6;
const D6: f64 = 5.475_938_084_995_344_946e-4;
const D7: f64 = 1.050_750_071_644_416_843_24e-9;

const E0: f64 = 6.657_904_643_501_103_777_2;
const E1: f64 = 5.463_784_911_164_114_369_9;
const E2: f64 = 1.784_826_539_917_291_335_8;
const E3: f64 = 0.296_560_571_828_504_891_23;
const E4: f64 = 0.026_532_189_526_576_123_093;
const E5: f64 = 0.001_242_660_947_388_078_438_6;
const E6: f64 = 2.711_555_568_743_487_578_15e-5;
const E7: f64 = 2.010_334_399_292_288_132_65e-7;

const F1: f64 = 0.599_832_206_555_887_937_69;
const F2: f64 = 0.136_929_880_922_735_805_31;
const F3: f64 = 0.014_875_361_290_850_614_852_5;
const F4: f64 = 0.001_369_298_809_227_358_053_1;
const F5: f64 = 1.964_319_203_976_612_415_24e-5;
const F6: f64 = 1.377_676_994_051_489_902_52e-7;

/// `numpy.percentile(a, q)` with the default `"linear"` interpolation.
///
/// `sorted` must already be ascending. `q` is in percent, matching numpy, so
/// `numpy.percentile(x, 97.5)` is `percentile(x, 97.5)`. A `NaN` quantile
/// propagates, which is how the bootstrap routines detect a degenerate
/// bias-correction and bail out.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 || q.is_nan() {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = (q / 100.0).clamp(0.0, 1.0) * (n as f64 - 1.0);
    let lo = pos.floor();
    let hi = pos.ceil();
    if lo == hi {
        sorted[lo as usize]
    } else {
        sorted[lo as usize] + (pos - lo) * (sorted[hi as usize] - sorted[lo as usize])
    }
}

/// Ordinary least squares fit of `y` on `x`, returning `(slope, intercept)`.
///
/// Mirrors `scipy.stats.linregress`'s first two return values.
pub fn linregress(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return None;
    }
    let x_mean = x.iter().sum::<f64>() / n as f64;
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - x_mean;
        sxx += dx * dx;
        sxy += dx * (yi - y_mean);
    }
    if sxx == 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    Some((slope, y_mean - slope * x_mean))
}

/// Mean of `x`, or `NaN` for an empty slice (as numpy does).
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        f64::NAN
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

/// Median of `x` (average of the two central order statistics for even `n`).
pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("bootstrap input must not contain NaN"));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Unbiased (`ddof=1`) sample variance, matching `numpy.var` defaults only when
/// `ddof=1` is passed. Returns `NaN` when there are fewer than two observations.
pub fn variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let x_mean = mean(x);
    let ss: f64 = x.iter().map(|v| (v - x_mean) * (v - x_mean)).sum();
    ss / (n as f64 - 1.0)
}

/// Sample standard deviation (`ddof=1`).
pub fn std(x: &[f64]) -> f64 {
    libm::sqrt(variance(x))
}

/// Geometric mean; returns `NaN` if any observation is non-positive.
pub fn geomean(x: &[f64]) -> f64 {
    if x.is_empty() || x.iter().any(|v| *v <= 0.0) {
        return f64::NAN;
    }
    libm::exp(x.iter().map(|v| libm::log(*v)).sum::<f64>() / x.len() as f64)
}

/// The reducers that can be bootstrapped, selected by name from Python.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Statistic {
    Mean,
    Median,
    Sum,
    Min,
    Max,
    Std,
    Var,
    Geomean,
}

impl Statistic {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "mean" => Some(Self::Mean),
            "median" => Some(Self::Median),
            "sum" => Some(Self::Sum),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "std" | "stdev" => Some(Self::Std),
            "var" | "variance" => Some(Self::Var),
            "geomean" | "geometric_mean" => Some(Self::Geomean),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Median => "median",
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::Std => "std",
            Self::Var => "var",
            Self::Geomean => "geomean",
        }
    }

    pub fn apply(self, x: &[f64]) -> f64 {
        match self {
            Self::Mean => mean(x),
            Self::Median => median(x),
            Self::Sum => x.iter().sum(),
            Self::Min => x.iter().copied().fold(f64::INFINITY, f64::min),
            Self::Max => x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Self::Std => std(x),
            Self::Var => variance(x),
            Self::Geomean => geomean(x),
        }
    }
}

/// Names of the supported statistics, for error messages.
pub const STATISTIC_NAMES: &[&str] = &[
    "mean", "median", "sum", "min", "max", "std", "var", "geomean",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Values cross-checked against `scipy.stats.norm`.
    #[test]
    fn norm_roundtrip() {
        for p in [1e-12, 1e-6, 0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999, 1.0 - 1e-9] {
            let z = norm_ppf(p);
            assert!((norm_cdf(z) - p).abs() < 1e-12, "p={p} z={z}");
        }
    }

    #[test]
    fn norm_ppf_known_values() {
        // scipy.stats.norm.ppf([0.025, 0.5, 0.975])
        assert!((norm_ppf(0.025) + 1.959_963_984_540_054).abs() < 1e-12);
        assert!((norm_ppf(0.5) - 0.0).abs() < 1e-15);
        assert!((norm_ppf(0.975) - 1.959_963_984_540_054).abs() < 1e-12);
    }

    #[test]
    fn norm_cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((norm_cdf(1.0) - 0.841_344_746_068_543).abs() < 1e-12);
        assert!((norm_cdf(-3.0) - 0.001_349_898_031_630_095).abs() < 1e-15);
    }

    #[test]
    fn percentile_matches_numpy() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&x, 0.0), 1.0);
        assert_eq!(percentile(&x, 100.0), 4.0);
        assert_eq!(percentile(&x, 50.0), 2.5);
        assert!((percentile(&x, 25.0) - 1.75).abs() < 1e-15);
    }

    #[test]
    fn linregress_known_line() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [3.0, 5.0, 7.0, 9.0];
        let (slope, intercept) = linregress(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-12);
        assert!((intercept - 1.0).abs() < 1e-12);
    }
}
