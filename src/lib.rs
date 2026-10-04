//! `polars_ros` -- Regression on Order Statistics and bootstrap confidence
//! intervals as native polars plugins.
//!
//! The heavy lifting lives in [`ros`] and [`bootstrap`]; this module only
//! exposes the compiled library to Python. The Python side lives in the
//! `polars_ros` package, which registers these functions through
//! `polars.plugins.register_plugin_function`.

mod bootstrap;
mod ros;
mod stats;
mod utils;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// Route Rust allocations through Python's allocator so arrow/polars buffers
/// and Python objects share one heap.
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
