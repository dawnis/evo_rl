//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;

mod population_api;

#[pyfunction]
fn log_something() {
    // This will use the logger installed in `my_module` to send the `info`
    // message to the Python logging facilities.
    info!("This is a test of pyo3-logging.");
}

/// A Python module for evo_rl implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn evo_rl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<population_api::PopulationApi>()?;
    let _ = m.add_function(wrap_pyfunction!(log_something, m)?);
    Ok(())
}

