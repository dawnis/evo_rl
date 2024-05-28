use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::ToPyObject;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use pyo3::types::PyDict;

/// Gene that defines the meta-learning rules for the neural network.
///
/// # Fields
/// * `innovation_number` - Unique identifier for this particular gene.
/// * `learning_rate` - Learning rate for synaptic adjustments.
/// * `learning_threshold` - Learning threshold for synaptic adjustments.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaLearningGene {
    pub innovation_number: Arc<str>,
    pub learning_rate: f32,
    pub learning_threshold: f32
}

impl Serialize for MetaLearningGene {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> 
    where 
        S: Serializer,
        {
            let mut state = serializer.serialize_struct("MetaLearningGene", 3)?;
            state.serialize_field("innovation_number", &self.innovation_number.as_ref())?;
            state.serialize_field("learning_rate", &self.learning_rate)?;
            state.serialize_field("learning_threshold", &self.learning_threshold)?;
            state.end()
        }
}

impl Default for MetaLearningGene {
    fn default() -> Self {
        Self {
            innovation_number: Arc::from("m01"),
            learning_rate: 0.001,
            learning_threshold: 0.5,
        }
    }
}

impl ToPyObject for MetaLearningGene {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("innnovation_number", &self.innovation_number.to_string());
        dict.set_item("learning_rate", self.learning_rate);
        dict.set_item("learning_threshold", self.learning_threshold);
        dict.into()
    }
}
