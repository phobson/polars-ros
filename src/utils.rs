//! Glue between polars `Series` and the plain `Vec`-based core algorithms.

use polars::prelude::*;
use polars_core::chunked_array::StructChunked;

/// Extract a `Float64` column as `Vec<Option<f64>>`, keeping nulls.
pub fn as_f64(s: &Series, name: &str) -> PolarsResult<Vec<Option<f64>>> {
    let ca = s.f64().map_err(|_| dtype_error(s, name, "Float64"))?;
    Ok(ca.iter().collect())
}

/// Extract a `Float64` column as `Vec<f64>`; nulls are an error.
///
/// For columns the algorithms index positionally, where a null would either
/// panic or silently shift the series.
pub fn as_f64_dense(s: &Series, name: &str) -> PolarsResult<Vec<f64>> {
    let ca = s.f64().map_err(|_| dtype_error(s, name, "Float64"))?;
    let mut out: Vec<f64> = Vec::with_capacity(ca.len());
    for v in ca.iter() {
        out.push(
            v.ok_or_else(|| polars_err!(ComputeError: "column `{name}` must not contain nulls"))?,
        );
    }
    Ok(out)
}

/// Extract a `Boolean` column as `Vec<bool>`; nulls count as `false` (a missing
/// qualifier means "not censored", which is the sensible default for
/// environmental data).
pub fn as_bool_lenient(s: &Series, name: &str) -> PolarsResult<Vec<bool>> {
    let ca = s.bool().map_err(|_| dtype_error(s, name, "Boolean"))?;
    Ok(ca.iter().map(|v| v.unwrap_or(false)).collect())
}

/// Extract any integer column as `Vec<usize>`, erroring on nulls and negatives.
///
/// Written with explicit loops rather than `collect`, because a plain `as usize`
/// cast would silently turn a negative `i32`/`i64` into an enormous index.
pub fn as_usize(s: &Series, name: &str) -> PolarsResult<Vec<usize>> {
    let null = || polars_err!(ComputeError: "column `{name}` must not contain nulls");
    let negative = || polars_err!(ComputeError: "column `{name}` must not contain negative values");

    let mut out: Vec<usize> = Vec::with_capacity(s.len());
    match s.dtype() {
        DataType::UInt32 => {
            for v in s.u32()?.iter() {
                out.push(v.ok_or_else(null)? as usize);
            }
        }
        DataType::UInt64 => {
            for v in s.u64()?.iter() {
                out.push(v.ok_or_else(null)? as usize);
            }
        }
        DataType::Int32 => {
            for v in s.i32()?.iter() {
                let v = v.ok_or_else(null)?;
                if v < 0 {
                    return Err(negative());
                }
                out.push(v as usize);
            }
        }
        DataType::Int64 => {
            for v in s.i64()?.iter() {
                let v = v.ok_or_else(null)?;
                if v < 0 {
                    return Err(negative());
                }
                out.push(v as usize);
            }
        }
        dtype => {
            return Err(polars_err!(SchemaMismatch:
                "expected `{name}` to be an integer dtype, got {dtype}"));
        }
    }
    Ok(out)
}

fn dtype_error(s: &Series, name: &str, expected: &str) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "expected `{name}` to have dtype {expected}, got {}", s.dtype()
    )
}

/// The Cohn numbers as they come back from a polars `Struct` series.
pub struct StructData {
    pub lower_dl: Vec<f64>,
    pub upper_dl: Vec<f64>,
    pub nuncen_above: Vec<f64>,
    pub nobs_below: Vec<f64>,
    pub ncen_equal: Vec<f64>,
    pub prob_exceedance: Vec<f64>,
}

impl StructData {
    pub fn from_series(s: &Series) -> PolarsResult<Self> {
        let ca = s.struct_()?;
        let field = |name: &str| -> PolarsResult<Vec<f64>> {
            let sub = ca.field_by_name(name)?;
            let sub = sub.f64().map_err(|_| {
                polars_err!(
                    SchemaMismatch: "Cohn number field `{name}` must be Float64, got {}",
                    sub.dtype()
                )
            })?;
            if sub.null_count() > 0 {
                polars_bail!(ComputeError: "Cohn number field `{name}` must not contain nulls");
            }
            Ok(sub.into_no_null_iter().collect())
        };
        Ok(Self {
            lower_dl: field("lower_dl")?,
            upper_dl: field("upper_dl")?,
            nuncen_above: field("nuncen_above")?,
            nobs_below: field("nobs_below")?,
            ncen_equal: field("ncen_equal")?,
            prob_exceedance: field("prob_exceedance")?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.lower_dl.is_empty()
    }
}

/// Assemble a `Struct` series from named `Float64` fields of equal length.
pub fn struct_series(name: &str, fields: Vec<(&str, Vec<f64>)>) -> PolarsResult<Series> {
    let series: Vec<Series> = fields
        .into_iter()
        .map(|(field, values)| {
            Float64Chunked::from_vec(PlSmallStr::from(field), values).into_series()
        })
        .collect();
    homogeneous_struct(name, 0, series)
}

/// Assemble a `Struct` series from already-built sub-series, each carrying its
/// own field name.
pub fn homogeneous_struct(name: &str, height: usize, series: Vec<Series>) -> PolarsResult<Series> {
    if series.is_empty() {
        polars_bail!(InvalidOperation: "cannot build a Struct series without fields");
    }
    let height = if height == 0 { series[0].len() } else { height };
    StructChunked::from_series(PlSmallStr::from(name), height, series.iter())
        .map(|ca| ca.into_series())
}

/// The polars `Field` describing a Cohn-number struct.
pub fn cohn_field(name: &str) -> Field {
    let inner = [
        "lower_dl",
        "upper_dl",
        "nuncen_above",
        "nobs_below",
        "ncen_equal",
        "prob_exceedance",
    ]
    .into_iter()
    .map(|f| Field::new(f.into(), DataType::Float64))
    .collect::<Vec<_>>();
    Field::new(name.into(), DataType::Struct(inner))
}
