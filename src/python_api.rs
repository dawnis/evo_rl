//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use pyo3::prelude::*;


//TODO: Population::new

#[pyclass]
/// Wrapper for Population
struct PopulationApi {
}

#[pymethods]
impl PopulationApi {
    #[new]
    fn new() -> Self {
        PopulationApi { }
    }
}

//TODO: population config
//TODO: Fitness eval loop


