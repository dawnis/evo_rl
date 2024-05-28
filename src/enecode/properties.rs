use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::ToPyObject;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use pyo3::types::PyDict;

/// Gene that defines the neuronal properties.
///
/// # Fields
/// * `innovation_number` - Unique identifier for this particular gene.
/// * `tau` - The time constant for the neuron.
/// * `homeostatic_force` - Homeostatic force for neuron.
/// * `tanh_alpha` - Scaling factor for tanh activation function.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronalPropertiesGene {
    pub innovation_number: Arc<str>,
    pub tau: f32,
    pub homeostatic_force: f32,
    pub tanh_alpha: f32,
}

impl Serialize for NeuronalPropertiesGene {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> 
    where 
        S: Serializer,
        {
            let mut state = serializer.serialize_struct("NeuronalPropertiesGene", 4)?;
            state.serialize_field("innovation_number", &self.innovation_number.as_ref())?;
            state.serialize_field("tau", &self.tau)?;
            state.serialize_field("tanh_alpha", &self.tanh_alpha)?;
            state.serialize_field("homeostatic_force", &self.homeostatic_force)?;
            state.end()
        }
}

impl Default for NeuronalPropertiesGene {
    fn default() -> Self {
        Self {
            innovation_number: Arc::from("p01"),
            tau: 0.,
            homeostatic_force: 0., 
            tanh_alpha: 1.
        }

    }
}

impl ToPyObject for NeuronalPropertiesGene {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("innovation_number", &self.innovation_number.to_string());
        dict.set_item("tau", self.tau);
        dict.set_item("homeostatic_force", self.homeostatic_force);
        dict.set_item("tanh_alpha", self.tanh_alpha);
        dict.into()
    }
}

