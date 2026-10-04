//! Regression on Order Statistics (ROS).
//!
//! A Rust port of the ROS half of the Python `wqio.ros` module. The core
//! routines in the first half of this file are plain `Vec`-in/`Vec`-out
//! functions so they can be unit tested without a `DataFrame`; the second half
//! wires them up as polars plugin expressions.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::str::FromStr;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::stats::{linregress, norm_ppf};
use crate::utils::*;

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// Monotonic transform applied before (`in`) or after (`out`) fitting the
/// regression of result on normal deviates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Transform {
    #[default]
    Identity,
    Log,
    Exp,
    Sqrt,
    Square,
}

impl FromStr for Transform {
    type Err = PolarsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "identity" | "none" => Ok(Self::Identity),
            "log" => Ok(Self::Log),
            "exp" => Ok(Self::Exp),
            "sqrt" => Ok(Self::Sqrt),
            "square" | "square_2" => Ok(Self::Square),
            other => polars_bail!(InvalidOperation:
                "unknown transform `{other}`, expected one of: identity, log, exp, sqrt, square"),
        }
    }
}

impl Transform {
    pub fn apply(self, v: f64) -> f64 {
        match self {
            Self::Identity => v,
            Self::Log => libm::log(v),
            Self::Exp => libm::exp(v),
            Self::Sqrt => libm::sqrt(v),
            Self::Square => v * v,
        }
    }
}

// ---------------------------------------------------------------------------
// Cohn numbers
// ---------------------------------------------------------------------------

/// The Cohn numbers, one row per unique detection limit.
///
/// wqio appends a padding row after the real ones so that
/// `prob_exceedance[i + 1]` stays in range for the last real `i`. That padding
/// row only ever holds `prob_exceedance == 0.0`, so reading it as "the
/// exceedance probability below the lowest detection limit" makes it natural to
/// leave out here; the remaining values are identical.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Cohn {
    pub lower_dl: Vec<f64>,
    pub upper_dl: Vec<f64>,
    pub nuncen_above: Vec<f64>,
    pub nobs_below: Vec<f64>,
    pub ncen_equal: Vec<f64>,
    pub prob_exceedance: Vec<f64>,
}

impl Cohn {
    pub fn is_empty(&self) -> bool {
        self.lower_dl.is_empty()
    }

    /// `prob_exceedance` for the row *below* row `i`; `0.0` past the end.
    fn pe_below(&self, i: usize) -> f64 {
        self.prob_exceedance.get(i + 1).copied().unwrap_or(0.0)
    }
}

/// Compute the Cohn numbers for a censored sample.
///
/// - `A_j` (`nuncen_above`): uncensored observations in `[lower_dl_j, upper_dl_j)`.
/// - `B_j` (`nobs_below`): observations below `lower_dl_j` (censored ones
///   counting as `<=`).
/// - `C_j` (`ncen_equal`): censored observations exactly at `lower_dl_j`.
///
/// Returns an empty [`Cohn`] when nothing is censored, matching
/// `wqio.ros.cohn_numbers`.
pub fn compute_cohn_numbers(result: &[Option<f64>], censored: &[bool]) -> Cohn {
    let mut lower_dl: Vec<f64> = Vec::new();

    // Unique detection limits, sorted ascending.
    let mut dls: Vec<f64> = Vec::new();
    for (value, is_censored) in result.iter().zip(censored) {
        if *is_censored {
            if let Some(v) = value {
                if !dls.contains(v) {
                    dls.push(*v);
                }
            }
        }
    }
    dls.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("detection limits must be comparable")
    });

    if !dls.is_empty() {
        // Anything below the smallest detection limit becomes its own row.
        if let Some(min_all) = result.iter().flatten().copied().reduce(f64::min) {
            if min_all < dls[0] {
                lower_dl.push(min_all);
            }
        }
        lower_dl.extend_from_slice(&dls);
    }

    let n = lower_dl.len();
    let mut cohn = Cohn {
        upper_dl: (0..n)
            .map(|j| lower_dl.get(j + 1).copied().unwrap_or(f64::INFINITY))
            .collect(),
        lower_dl,
        ..Default::default()
    };

    for j in 0..n {
        let lo = cohn.lower_dl[j];
        let hi = cohn.upper_dl[j];

        cohn.nuncen_above.push(
            result
                .iter()
                .zip(censored)
                .filter(|(v, c)| !**c && v.is_some_and(|v| v >= lo && v < hi))
                .count() as f64,
        );
        cohn.nobs_below.push(
            result
                .iter()
                .zip(censored)
                .filter(|(v, c)| match v {
                    Some(v) => {
                        if **c {
                            *v <= lo
                        } else {
                            *v < lo
                        }
                    }
                    None => false,
                })
                .count() as f64,
        );
        cohn.ncen_equal.push(
            result
                .iter()
                .zip(censored)
                .filter(|(v, c)| **c && v.is_some_and(|v| v == lo))
                .count() as f64,
        );
    }

    // P(exceed) by backward recursion, with P = 0 past the last row.
    let mut pe = vec![0.0; n + 1];
    for j in (0..n).rev() {
        let a = cohn.nuncen_above[j];
        let b = cohn.nobs_below[j];
        pe[j] = pe[j + 1] + (1.0 - pe[j + 1]) * a / (a + b);
    }
    pe.truncate(n);
    cohn.prob_exceedance = pe;

    cohn
}

/// Index of the Cohn row a single observation falls into: the last row whose
/// `lower_dl` is `<= value`.
pub fn detection_limit_index(value: Option<f64>, lower_dl: &[f64]) -> PolarsResult<usize> {
    let Some(value) = value else {
        polars_bail!(InvalidOperation: "cannot compute a detection limit index for a null result");
    };
    let mut idx = None;
    for (j, lower) in lower_dl.iter().enumerate() {
        if *lower <= value {
            idx = Some(j);
        } else {
            break;
        }
    }
    idx.ok_or_else(|| {
        polars_err!(ComputeError:
            "observation {value} falls below every detection limit, so ROS cannot place it")
    })
}

/// Map every observation onto its Cohn row.
pub fn compute_detection_limit_indices(
    result: &[Option<f64>],
    lower_dl: &[f64],
) -> PolarsResult<Vec<usize>> {
    result
        .iter()
        .map(|v| detection_limit_index(*v, lower_dl))
        .collect()
}

/// 1-based running count within each `(detection limit index, censored)` group.
pub fn compute_group_rank(dl_idx: &[usize], censored: &[bool]) -> Vec<u32> {
    let mut counts: HashMap<(usize, bool), u32> = HashMap::new();
    let mut ranks = Vec::with_capacity(dl_idx.len());
    for (&i, &c) in dl_idx.iter().zip(censored) {
        let counter = counts.entry((i, c)).or_insert(0);
        *counter += 1;
        ranks.push(*counter);
    }
    ranks
}

/// ROS plotting positions for every observation.
///
/// Censored observations get `(1 - PE_j) * rank / (C_j + 1)`; uncensored ones
/// get `(1 - PE_j) + (PE_j - PE_{j+1}) * rank / (A_j + 1)`. Finally the
/// censored plotting positions are sorted ascending among themselves, which is
/// what makes the resulting probability plot monotone.
pub fn compute_plotting_positions(
    result: &[Option<f64>],
    censored: &[bool],
    cohn: &Cohn,
) -> PolarsResult<Vec<f64>> {
    let dl_idx = compute_detection_limit_indices(result, &cohn.lower_dl)?;
    let ranks = compute_group_rank(&dl_idx, censored);

    let mut positions: Vec<f64> = Vec::with_capacity(result.len());
    for ((&i, &rank), &is_censored) in dl_idx.iter().zip(&ranks).zip(censored) {
        let pe_here = cohn.prob_exceedance[i];
        if is_censored {
            positions.push((1.0 - pe_here) * rank as f64 / (cohn.ncen_equal[i] + 1.0));
        } else {
            positions.push(
                (1.0 - pe_here)
                    + (pe_here - cohn.pe_below(i)) * rank as f64 / (cohn.nuncen_above[i] + 1.0),
            );
        }
    }

    let mut nd: Vec<f64> = censored
        .iter()
        .zip(&positions)
        .filter(|(c, _)| **c)
        .map(|(_, p)| *p)
        .collect();
    nd.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("plotting positions must be comparable")
    });
    let mut nd_iter = nd.into_iter();
    for (is_censored, slot) in censored.iter().zip(positions.iter_mut()) {
        if *is_censored {
            *slot = nd_iter
                .next()
                .expect("one plotting position per censored row");
        }
    }

    Ok(positions)
}

/// Convert plotting positions into standard normal deviates.
pub fn compute_zprelim(plot_pos: &[f64]) -> Vec<f64> {
    plot_pos.iter().map(|p| norm_ppf(*p)).collect()
}

// ---------------------------------------------------------------------------
// Imputation
// ---------------------------------------------------------------------------

/// Tuning knobs for [`impute`], mirroring `wqio.ros.ROS`.
#[derive(Clone, Copy, Debug)]
pub struct ImputeOptions {
    pub min_uncensored: usize,
    pub max_fraction_censored: f64,
    pub substitution_fraction: f64,
    pub transform_in: Transform,
    pub transform_out: Transform,
    pub floor: Option<f64>,
}

impl Default for ImputeOptions {
    fn default() -> Self {
        Self {
            min_uncensored: 2,
            max_fraction_censored: 0.8,
            substitution_fraction: 0.5,
            transform_in: Transform::Log,
            transform_out: Transform::Exp,
            floor: None,
        }
    }
}

/// `(enough_uncensored, not_too_many_censored)`; ROS runs only if both hold.
pub fn validity(censored: &[bool], opts: &ImputeOptions) -> (bool, bool) {
    let n = censored.len();
    if n == 0 {
        return (false, false);
    }
    let n_censored = censored.iter().filter(|c| **c).count();
    let fraction_censored = n_censored as f64 / n as f64;
    (
        n - n_censored >= opts.min_uncensored,
        fraction_censored <= opts.max_fraction_censored,
    )
}

fn cmp_optional_f64(a: Option<f64>, b: Option<f64>) -> Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
    }
}

/// Everything [`do_ros`] produces, in ROS order.
#[derive(Clone, Debug)]
pub struct RosDetail {
    /// Original row index for each ROS-ordered row.
    pub order: Vec<usize>,
    pub result: Vec<f64>,
    pub censored: Vec<bool>,
    pub det_limit_index: Vec<usize>,
    pub rank: Vec<u32>,
    pub plot_pos: Vec<f64>,
    pub zprelim: Vec<f64>,
    pub estimated: Vec<f64>,
    pub final_values: Vec<f64>,
}

/// Full ROS pass: sort, build Cohn numbers, rank, plot, fit and impute.
///
/// The returned vectors are in ROS order (censored first, ascending by value).
/// [`impute`] re-scatters them onto the caller's original row order. Rows with a
/// missing result are omitted, as are censored results above the maximum
/// uncensored one.
pub fn do_ros(
    result: &[Option<f64>],
    censored: &[bool],
    opts: &ImputeOptions,
) -> PolarsResult<RosDetail> {
    let cohn = compute_cohn_numbers(result, censored);
    if cohn.is_empty() {
        polars_bail!(InvalidOperation:
            "ROS needs at least one censored observation, but every value is uncensored");
    }

    // Censored observations above the largest uncensored one cannot be placed on
    // the probability plot, so they are dropped (wqio logs a warning here).
    let max_uncensored = result
        .iter()
        .zip(censored)
        .filter(|(_, c)| !**c)
        .filter_map(|(v, _)| *v)
        .reduce(f64::max);

    let mut order: Vec<usize> = (0..result.len()).collect();
    // Descending on `censored`, ascending on `result`, nulls last, stable.
    order.sort_by(|&a, &b| {
        censored[b]
            .cmp(&censored[a])
            .then_with(|| cmp_optional_f64(result[a], result[b]))
    });
    order.retain(|&i| {
        // A missing result carries no information, so it cannot be placed on the
        // probability plot. Dropping it here makes it come back as a null.
        if result[i].is_none() {
            return false;
        }
        // Likewise for censored observations above the largest uncensored one.
        !censored[i] || result[i].is_some_and(|v| max_uncensored.is_some_and(|m| v <= m))
    });

    let sorted_result: Vec<Option<f64>> = order.iter().map(|&i| result[i]).collect();
    let sorted_censored: Vec<bool> = order.iter().map(|&i| censored[i]).collect();

    let det_limit_index = compute_detection_limit_indices(&sorted_result, &cohn.lower_dl)?;
    let rank = compute_group_rank(&det_limit_index, &sorted_censored);
    let plot_pos = compute_plotting_positions(&sorted_result, &sorted_censored, &cohn)?;
    let zprelim = compute_zprelim(&plot_pos);

    let (estimated, final_values) = ros_estimate(
        &zprelim,
        &sorted_result,
        &sorted_censored,
        opts.transform_in,
        opts.transform_out,
    )?;

    Ok(RosDetail {
        order,
        result: sorted_result
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect(),
        censored: sorted_censored,
        det_limit_index,
        rank,
        plot_pos,
        zprelim,
        estimated,
        final_values,
    })
}

/// Fit a line to `transform_in(result)` against the normal deviates of the
/// *uncensored* observations, then invert it for the censored ones.
///
/// Returns `(estimated, final)`, where `estimated` is only defined on censored
/// rows and `final` holds the original observations everywhere else.
pub fn ros_estimate(
    zprelim: &[f64],
    result: &[Option<f64>],
    censored: &[bool],
    transform_in: Transform,
    transform_out: Transform,
) -> PolarsResult<(Vec<f64>, Vec<f64>)> {
    let (xs, ys): (Vec<f64>, Vec<f64>) = zprelim
        .iter()
        .zip(result)
        .zip(censored)
        .filter(|(_, c)| !**c)
        .map(|((&z, v), _)| (z, transform_in.apply(v.unwrap_or(f64::NAN))))
        .unzip();

    let (slope, intercept) = linregress(&xs, &ys).ok_or_else(|| {
        polars_err!(ComputeError:
            "the uncensored observations have no spread in their plotting positions, \
             so a regression on order statistics cannot be fitted")
    })?;

    let mut estimated = vec![f64::NAN; result.len()];
    let mut final_values = Vec::with_capacity(result.len());
    for i in 0..result.len() {
        if censored[i] {
            let e = transform_out.apply(slope * zprelim[i] + intercept);
            estimated[i] = e;
            final_values.push(e);
        } else {
            final_values.push(result[i].unwrap_or(f64::NAN));
        }
    }

    Ok((estimated, final_values))
}

/// Impute censored observations with ROS, falling back to simple substitution.
///
/// The returned vector is aligned with the input rows. Observations ROS has to
/// drop -- missing results, and censored results above the maximum uncensored
/// value -- come back as `None`, since there is no defensible imputation for
/// them.
///
/// * nothing censored -> the input, unchanged;
/// * fewer than `min_uncensored` uncensored, or more than
///   `max_fraction_censored` censored -> substitution;
/// * otherwise -> regression on order statistics.
pub fn impute(
    result: &[Option<f64>],
    censored: &[bool],
    opts: &ImputeOptions,
) -> PolarsResult<Vec<Option<f64>>> {
    let n_censored = censored.iter().filter(|c| **c).count();
    if n_censored == 0 {
        return Ok(result.to_vec());
    }

    let (enough_uncensored, not_too_many_censored) = validity(censored, opts);
    if !(enough_uncensored && not_too_many_censored) {
        return Ok(substitute(result, censored, opts));
    }

    let detail = do_ros(result, censored, opts)?;
    let mut out: Vec<Option<f64>> = vec![None; result.len()];
    for (&original, &value) in detail.order.iter().zip(&detail.final_values) {
        out[original] = Some(apply_floor(value, opts.floor));
    }
    Ok(out)
}

fn apply_floor(value: f64, floor: Option<f64>) -> f64 {
    match floor {
        Some(floor) if value < floor => floor,
        _ => value,
    }
}

/// Simple substitution: censored values become a fraction of their detection
/// limit, everything else is left alone.
pub fn substitute(
    result: &[Option<f64>],
    censored: &[bool],
    opts: &ImputeOptions,
) -> Vec<Option<f64>> {
    result
        .iter()
        .zip(censored)
        .map(|(v, c)| {
            v.map(|v| {
                let v = if *c {
                    v * opts.substitution_fraction
                } else {
                    v
                };
                apply_floor(v, opts.floor)
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Plugin kwargs
// ---------------------------------------------------------------------------

fn default_min_uncensored() -> usize {
    2
}

fn default_max_fraction_censored() -> f64 {
    0.8
}

fn default_substitution_fraction() -> f64 {
    0.5
}

fn default_transform_in() -> String {
    "log".to_string()
}

fn default_transform_out() -> String {
    "exp".to_string()
}

#[derive(Debug, Deserialize)]
struct TransformKwargs {
    #[serde(default = "default_transform_in")]
    transform_in: String,
    #[serde(default = "default_transform_out")]
    transform_out: String,
}

impl Default for TransformKwargs {
    fn default() -> Self {
        Self {
            transform_in: default_transform_in(),
            transform_out: default_transform_out(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ImputeKwargs {
    #[serde(default = "default_min_uncensored")]
    min_uncensored: usize,
    #[serde(default = "default_max_fraction_censored")]
    max_fraction_censored: f64,
    #[serde(default = "default_substitution_fraction")]
    substitution_fraction: f64,
    #[serde(default = "default_transform_in")]
    transform_in: String,
    #[serde(default = "default_transform_out")]
    transform_out: String,
    #[serde(default)]
    floor: Option<f64>,
}

impl Default for ImputeKwargs {
    fn default() -> Self {
        Self {
            min_uncensored: default_min_uncensored(),
            max_fraction_censored: default_max_fraction_censored(),
            substitution_fraction: default_substitution_fraction(),
            transform_in: default_transform_in(),
            transform_out: default_transform_out(),
            floor: None,
        }
    }
}

impl ImputeKwargs {
    fn options(&self) -> PolarsResult<ImputeOptions> {
        Ok(ImputeOptions {
            min_uncensored: self.min_uncensored,
            max_fraction_censored: self.max_fraction_censored,
            substitution_fraction: self.substitution_fraction,
            transform_in: self.transform_in.parse()?,
            transform_out: self.transform_out.parse()?,
            floor: self.floor,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ValidityKwargs {
    #[serde(default = "default_min_uncensored")]
    min_uncensored: usize,
    #[serde(default = "default_max_fraction_censored")]
    max_fraction_censored: f64,
}

impl Default for ValidityKwargs {
    fn default() -> Self {
        Self {
            min_uncensored: default_min_uncensored(),
            max_fraction_censored: default_max_fraction_censored(),
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin expressions
// ---------------------------------------------------------------------------

fn check_same_len(inputs: &[Series]) -> PolarsResult<()> {
    let n = inputs[0].len();
    if inputs.iter().any(|s| s.len() != n) {
        polars_bail!(ShapeMismatch: "all inputs must have the same length");
    }
    Ok(())
}

fn cohn_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(cohn_field("cohn_numbers"))
}

fn detail_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let inner = [
        ("result", DataType::Float64),
        ("censored", DataType::Boolean),
        ("det_limit_index", DataType::UInt32),
        ("rank", DataType::UInt32),
        ("plot_pos", DataType::Float64),
        ("zprelim", DataType::Float64),
        ("estimated", DataType::Float64),
        ("final", DataType::Float64),
    ]
    .into_iter()
    .map(|(name, dtype)| Field::new(name.into(), dtype))
    .collect::<Vec<_>>();
    Ok(Field::new("ros_detail".into(), DataType::Struct(inner)))
}

fn unpack_cohn(s: &Series) -> PolarsResult<Cohn> {
    let data = StructData::from_series(s)?;
    Ok(Cohn {
        lower_dl: data.lower_dl,
        upper_dl: data.upper_dl,
        nuncen_above: data.nuncen_above,
        nobs_below: data.nobs_below,
        ncen_equal: data.ncen_equal,
        prob_exceedance: data.prob_exceedance,
    })
}

/// `pl.ros.cohn_numbers(result, censorship)`
#[polars_expr(output_type_func = cohn_output_type)]
fn cohn_numbers(inputs: &[Series]) -> PolarsResult<Series> {
    check_same_len(inputs)?;
    let result = as_f64(&inputs[0], "result")?;
    let censored = as_bool_lenient(&inputs[1], "censorship")?;
    let cohn = compute_cohn_numbers(&result, &censored);
    struct_series(
        "cohn_numbers",
        vec![
            ("lower_dl", cohn.lower_dl),
            ("upper_dl", cohn.upper_dl),
            ("nuncen_above", cohn.nuncen_above),
            ("nobs_below", cohn.nobs_below),
            ("ncen_equal", cohn.ncen_equal),
            ("prob_exceedance", cohn.prob_exceedance),
        ],
    )
}

/// `pl.ros.detection_limit_index(result, cohn)`
#[polars_expr(output_type=UInt32)]
fn detection_limit_index(inputs: &[Series]) -> PolarsResult<Series> {
    let result = as_f64(&inputs[0], "result")?;
    let data = StructData::from_series(&inputs[1])?;
    if data.is_empty() {
        polars_bail!(InvalidOperation:
            "detection limit indices need at least one censored observation, \
             but the Cohn numbers are empty");
    }
    let indices = compute_detection_limit_indices(&result, &data.lower_dl)?;
    let indices: Vec<u32> = indices.into_iter().map(|i| i as u32).collect();
    Ok(UInt32Chunked::from_slice(inputs[0].name().clone(), &indices).into_series())
}

/// `pl.ros.group_rank(det_limit_index, censorship)`
#[polars_expr(output_type=UInt32)]
fn group_rank(inputs: &[Series]) -> PolarsResult<Series> {
    check_same_len(inputs)?;
    let dl_idx = as_usize(&inputs[0], "det_limit_index")?;
    let censored = as_bool_lenient(&inputs[1], "censorship")?;
    let ranks = compute_group_rank(&dl_idx, &censored);
    Ok(UInt32Chunked::from_slice(inputs[1].name().clone(), &ranks).into_series())
}

/// `pl.ros.plotting_positions(result, censorship, cohn)`
#[polars_expr(output_type=Float64)]
fn plotting_positions(inputs: &[Series]) -> PolarsResult<Series> {
    check_same_len(&inputs[..2])?;
    let result = as_f64(&inputs[0], "result")?;
    let censored = as_bool_lenient(&inputs[1], "censorship")?;
    let cohn = unpack_cohn(&inputs[2])?;
    let positions = compute_plotting_positions(&result, &censored, &cohn)?;
    Ok(Float64Chunked::from_vec(inputs[0].name().clone(), positions).into_series())
}

/// `pl.ros.zprelim(plotting_positions)`
#[polars_expr(output_type=Float64)]
fn zprelim(inputs: &[Series]) -> PolarsResult<Series> {
    let plot_pos = as_f64_dense(&inputs[0], "plotting_positions")?;
    Ok(
        Float64Chunked::from_vec(inputs[0].name().clone(), compute_zprelim(&plot_pos))
            .into_series(),
    )
}

/// `pl.ros.estimate(zprelim, result, censorship)`
#[polars_expr(output_type=Float64)]
fn estimate(inputs: &[Series], kwargs: TransformKwargs) -> PolarsResult<Series> {
    check_same_len(inputs)?;
    let z = as_f64_dense(&inputs[0], "zprelim")?;
    let result = as_f64(&inputs[1], "result")?;
    let censored = as_bool_lenient(&inputs[2], "censorship")?;
    let (_, final_values) = ros_estimate(
        &z,
        &result,
        &censored,
        kwargs.transform_in.parse()?,
        kwargs.transform_out.parse()?,
    )?;
    Ok(Float64Chunked::from_vec(inputs[1].name().clone(), final_values).into_series())
}

/// `pl.ros.is_valid(censorship)`
#[polars_expr(output_type=Boolean)]
fn is_valid(inputs: &[Series], kwargs: ValidityKwargs) -> PolarsResult<Series> {
    let censored = as_bool_lenient(&inputs[0], "censorship")?;
    let opts = ImputeOptions {
        min_uncensored: kwargs.min_uncensored,
        max_fraction_censored: kwargs.max_fraction_censored,
        ..Default::default()
    };
    let (enough, not_too_many) = validity(&censored, &opts);
    Ok(
        BooleanChunked::from_slice(inputs[0].name().clone(), &[enough && not_too_many])
            .into_series(),
    )
}

/// `pl.ros.impute(result, censorship)`
#[polars_expr(output_type=Float64)]
fn impute_expr(inputs: &[Series], kwargs: ImputeKwargs) -> PolarsResult<Series> {
    check_same_len(inputs)?;
    let result = as_f64(&inputs[0], "result")?;
    let censored = as_bool_lenient(&inputs[1], "censorship")?;
    let values = impute(&result, &censored, &kwargs.options()?)?;
    Ok(
        Float64Chunked::from_iter_options(inputs[0].name().clone(), values.into_iter())
            .into_series(),
    )
}

/// `pl.ros.substitute(result, censorship)` -- the simple-substitution fallback
/// on its own.
#[polars_expr(output_type=Float64)]
fn substitute_expr(inputs: &[Series], kwargs: ImputeKwargs) -> PolarsResult<Series> {
    check_same_len(inputs)?;
    let result = as_f64(&inputs[0], "result")?;
    let censored = as_bool_lenient(&inputs[1], "censorship")?;
    let values = substitute(&result, &censored, &kwargs.options()?);
    Ok(
        Float64Chunked::from_iter_options(inputs[0].name().clone(), values.into_iter())
            .into_series(),
    )
}

/// `pl.ros.impute_details(result, censorship)` -- every ROS intermediate column,
/// in ROS order.
#[polars_expr(output_type_func = detail_output_type)]
fn impute_details(inputs: &[Series], kwargs: ImputeKwargs) -> PolarsResult<Series> {
    check_same_len(inputs)?;
    let result = as_f64(&inputs[0], "result")?;
    let censored = as_bool_lenient(&inputs[1], "censorship")?;
    let opts = kwargs.options()?;
    let detail = do_ros(&result, &censored, &opts)?;
    let height = detail.order.len();
    let final_values: Vec<f64> = detail
        .final_values
        .iter()
        .map(|v| apply_floor(*v, opts.floor))
        .collect();

    homogeneous_struct(
        "ros_detail",
        height,
        vec![
            Float64Chunked::from_vec("result".into(), detail.result).into_series(),
            BooleanChunked::from_slice("censored".into(), &detail.censored).into_series(),
            UInt32Chunked::from_iter_options(
                "det_limit_index".into(),
                detail.det_limit_index.iter().map(|i| Some(*i as u32)),
            )
            .into_series(),
            UInt32Chunked::from_slice("rank".into(), &detail.rank).into_series(),
            Float64Chunked::from_vec("plot_pos".into(), detail.plot_pos).into_series(),
            Float64Chunked::from_vec("zprelim".into(), detail.zprelim).into_series(),
            Float64Chunked::from_vec("estimated".into(), detail.estimated).into_series(),
            Float64Chunked::from_slice("final".into(), &final_values).into_series(),
        ],
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The 35-row fixture from `wqio.tests.test_ros`.
    fn test_ros_data() -> (Vec<Option<f64>>, Vec<bool>) {
        let raw = [
            (2.00, false),
            (4.20, false),
            (4.62, false),
            (5.00, true),
            (5.00, true),
            (5.50, true),
            (5.57, false),
            (5.66, false),
            (5.75, true),
            (5.86, false),
            (6.65, false),
            (6.78, false),
            (6.79, false),
            (7.50, false),
            (7.50, false),
            (7.50, false),
            (8.63, false),
            (8.71, false),
            (8.99, false),
            (9.50, true),
            (9.50, true),
            (9.85, false),
            (10.82, false),
            (11.00, true),
            (11.25, false),
            (11.25, false),
            (12.20, false),
            (14.92, false),
            (16.77, false),
            (17.81, false),
            (19.16, false),
            (19.19, false),
            (19.64, false),
            (20.18, false),
            (22.97, false),
        ];
        (
            raw.iter().map(|(v, _)| Some(*v)).collect(),
            raw.iter().map(|(_, c)| *c).collect(),
        )
    }

    #[test]
    fn cohn_numbers_match_wqio_fixture() {
        let (result, censored) = test_ros_data();
        let cohn = compute_cohn_numbers(&result, &censored);

        assert_eq!(cohn.lower_dl, [2.0, 5.0, 5.5, 5.75, 9.5, 11.0]);
        assert_eq!(cohn.upper_dl, [5.0, 5.5, 5.75, 9.5, 11.0, f64::INFINITY]);
        assert_eq!(cohn.nuncen_above, [3.0, 0.0, 2.0, 10.0, 2.0, 11.0]);
        assert_eq!(cohn.nobs_below, [0.0, 5.0, 6.0, 9.0, 21.0, 24.0]);
        assert_eq!(cohn.ncen_equal, [0.0, 2.0, 1.0, 1.0, 2.0, 1.0]);

        let expected_pe = [
            1.0,
            0.777_574_370_709_382_2,
            0.777_574_370_709_382_2,
            0.703_432_494_279_176_2,
            0.373_913_043_478_260_9,
            0.314_285_714_285_714_3,
        ];
        for (got, want) in cohn.prob_exceedance.iter().zip(expected_pe) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
    }

    #[test]
    fn cohn_numbers_empty_without_censored() {
        let result: Vec<Option<f64>> = [1.0, 2.0, 3.0].into_iter().map(Some).collect();
        assert!(compute_cohn_numbers(&result, &[false, false, false]).is_empty());
    }

    #[test]
    fn detection_limit_indices_match_wqio_fixture() {
        let (result, censored) = test_ros_data();
        let cohn = compute_cohn_numbers(&result, &censored);
        assert_eq!(detection_limit_index(Some(3.5), &cohn.lower_dl).unwrap(), 0);
        assert_eq!(detection_limit_index(Some(6.0), &cohn.lower_dl).unwrap(), 3);
        assert_eq!(
            detection_limit_index(Some(12.0), &cohn.lower_dl).unwrap(),
            5
        );
        assert!(detection_limit_index(Some(0.0), &cohn.lower_dl).is_err());
    }

    #[test]
    fn group_rank_matches_wqio_semantics() {
        // wqio ranks within groups of (detection_limit_index, censorship) on the
        // ROS-sorted frame -- see the `_ros_group_rank` call in `wqio.ros`. Note
        // wqio's own unit test for this passes a categorical string in place of
        // the censorship column, so it exercises a different grouping; this
        // fixture pins the behaviour the pipeline actually depends on.
        let dl_idx = vec![0usize, 0, 0, 1, 1, 1];
        let censored = vec![false, false, true, true, true, false];
        assert_eq!(
            compute_group_rank(&dl_idx, &censored),
            vec![1, 2, 1, 1, 2, 1]
        );
    }

    #[test]
    fn plotting_positions_match_wqio_fixture() {
        let (result, censored) = test_ros_data();
        let cohn = compute_cohn_numbers(&result, &censored);
        let positions = compute_plotting_positions(&result, &censored, &cohn).unwrap();
        // Expected values from `wqio.tests.test_ros.test_plotting_positions`,
        // reordered from ROS order into the original row order.
        let expected = [
            0.05560641, 0.11121281, 0.16681922, 0.07414188, 0.11121281, 0.14828375, 0.24713959,
            0.27185355, 0.14828375, 0.32652382, 0.35648013, 0.38643645, 0.41639276, 0.44634907,
            0.47630539, 0.50626170, 0.53621802, 0.56617433, 0.59613064, 0.20869565, 0.34285714,
            0.64596273, 0.66583851, 0.41739130, 0.71190476, 0.73809524, 0.76428571, 0.79047619,
            0.81666667, 0.84285714, 0.86904762, 0.89523810, 0.92142857, 0.94761905, 0.97380952,
        ];
        for (got, want) in positions.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-7,
                "plotting position mismatch: {got} != {want}"
            );
        }
    }

    #[test]
    fn impute_matches_wqio_baseline() {
        let (result, censored) = test_ros_data();
        let mut values: Vec<f64> = impute(&result, &censored, &ImputeOptions::default())
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        // wqio returns ROS order (censored block, then uncensored); this port
        // keeps input row order, so compare the sorted multisets.
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected = vec![
            3.11279729, 3.60634338, 4.04602788, 4.04602788, 4.71008116, 6.14010906, 6.97841457,
            2.0, 4.2, 4.62, 5.57, 5.66, 5.86, 6.65, 6.78, 6.79, 7.5, 7.5, 7.5, 8.63, 8.71, 8.99,
            9.85, 10.82, 11.25, 11.25, 12.2, 14.92, 16.77, 17.81, 19.16, 19.19, 19.64, 20.18,
            22.97,
        ];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values.len(), expected.len());
        for (got, want) in values.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
    }

    #[test]
    fn impute_with_floor_matches_wqio() {
        let (result, censored) = test_ros_data();
        let opts = ImputeOptions {
            floor: Some(5.0),
            ..Default::default()
        };
        let mut values: Vec<f64> = impute(&result, &censored, &opts)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected = vec![
            5.0, 5.0, 5.0, 5.0, 5.0, 6.14010906, 6.97841457, 5.0, 5.0, 5.0, 5.57, 5.66, 5.86, 6.65,
            6.78, 6.79, 7.5, 7.5, 7.5, 8.63, 8.71, 8.99, 9.85, 10.82, 11.25, 11.25, 12.2, 14.92,
            16.77, 17.81, 19.16, 19.19, 19.64, 20.18, 22.97,
        ];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values.len(), expected.len());
        for (got, want) in values.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
    }

    #[test]
    fn all_equal_values_are_left_alone() {
        let result: Vec<Option<f64>> = vec![Some(0.4); 8];
        let censored = vec![true, true, true, true, true, false, true, false];
        let values: Vec<f64> = impute(&result, &censored, &ImputeOptions::default())
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert!(values.iter().all(|v| (v - 0.4).abs() < 1e-12));
    }

    #[test]
    fn one_uncensored_observation_falls_back_to_substitution() {
        let result: Vec<Option<f64>> = [Some(1.0), Some(1.0), Some(12.0), Some(15.0)]
            .into_iter()
            .collect();
        let censored = vec![true, true, true, false];
        let values: Vec<f64> = impute(&result, &censored, &ImputeOptions::default())
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(values, vec![0.5, 0.5, 6.0, 15.0]);
    }

    #[test]
    fn too_many_censored_falls_back_to_substitution() {
        let result: Vec<Option<f64>> = (0..18).map(|i| Some(i as f64)).collect();
        let mut censored = vec![true; 15];
        censored.extend_from_slice(&[false, false, false]);
        let values: Vec<f64> = impute(&result, &censored, &ImputeOptions::default())
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[15], 15.0);
    }

    #[test]
    fn no_censored_values_are_a_no_op() {
        let result: Vec<Option<f64>> = [Some(1.0), Some(2.0)].into_iter().collect();
        assert_eq!(
            impute(&result, &[false, false], &ImputeOptions::default()).unwrap(),
            result
        );
    }

    #[test]
    fn censored_above_max_uncensored_become_null() {
        let result: Vec<Option<f64>> = [Some(1.0), Some(10.0), Some(5.0), Some(60.0)]
            .into_iter()
            .collect();
        let censored = vec![true, false, false, true];
        let imputed = impute(&result, &censored, &ImputeOptions::default()).unwrap();
        assert!(imputed[3].is_none());
        assert!(imputed.iter().take(3).all(Option::is_some));
    }

    #[test]
    fn missing_results_propagate_as_null() {
        let result: Vec<Option<f64>> = vec![Some(1.0), None, Some(3.0), Some(4.0)];
        let censored = vec![true, false, false, false];
        let imputed = impute(&result, &censored, &ImputeOptions::default()).unwrap();
        assert!(imputed[1].is_none());
        assert_eq!(imputed[2], Some(3.0));
        assert_eq!(imputed[3], Some(4.0));
        assert!(imputed[0].is_some());
    }

    #[test]
    fn a_missing_uncensored_result_does_not_break_ros() {
        // The uncensored null has to be dropped before plotting positions are
        // computed, otherwise the regression has no observations to fit.
        let result: Vec<Option<f64>> = vec![Some(1.0), None, Some(3.0), Some(4.0)];
        let censored = vec![true, false, false, false];
        let detail = do_ros(&result, &censored, &ImputeOptions::default()).unwrap();
        assert_eq!(detail.order.len(), 3);
        assert!(!detail.order.contains(&1));
    }

    #[test]
    fn a_missing_censored_result_propagates_as_null() {
        // Two uncensored values, so this takes the ROS path rather than the
        // substitution fallback.
        let result: Vec<Option<f64>> = vec![Some(1.0), None, Some(3.0), Some(4.0)];
        let censored = vec![true, true, false, false];
        let imputed = impute(&result, &censored, &ImputeOptions::default()).unwrap();
        assert!(imputed[1].is_none());
        assert_eq!(imputed[2], Some(3.0));
        assert_eq!(imputed[3], Some(4.0));
        assert!(imputed[0].is_some());
    }
}
