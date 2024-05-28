use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::ToPyObject;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use std::collections::HashMap;
use pyo3::types::PyDict;
use crate::enecode::NeuronType;

/// Gene that defines the topology of a neuron.
///
/// # Fields
/// * `innovation_number` - Unique identifier for this particular topology.
/// * `pin` - The type of neuron (Input, Output, Hidden).
/// * `inputs` - The identifiers for input neurons.
/// * `outputs` - The identifiers for output neurons.
/// * `genetic_weights` - Weights for each of the input neurons.
/// * `genetic_bias` - The bias term for the neuron.
/// * `active` - Whether the neuron is currently active.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyGene {
    pub innovation_number: Arc<str>,
    pub pin: NeuronType, //stolen from python-neat for outside connections
    pub inputs: HashMap<String, f32>, //map that defines genetic weight of synapse for each parent
    pub genetic_bias: f32,
    pub active: bool,
}

impl Serialize for TopologyGene {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> 
    where 
        S: Serializer,
        {
            let mut state = serializer.serialize_struct("TopologyGene", 5)?;
            state.serialize_field("innovation_number", &self.innovation_number.as_ref())?;
            state.serialize_field("pin", &self.pin)?;
            state.serialize_field("inputs", &self.inputs)?;
            state.serialize_field("genetic_bias", &self.genetic_bias)?;
            state.serialize_field("active", &self.active)?;
            state.end()
        }
}

impl ToPyObject for TopologyGene {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("innovation_number", &self.innovation_number.to_string());

        match self.pin {
            NeuronType::In => dict.set_item("pin", "Input"),
            NeuronType::Out => dict.set_item("pin", "Output"),
            NeuronType::Hidden => dict.set_item("pin", "Hidden"),
        };

        dict.set_item("inputs", &self.inputs);
        dict.set_item("genetic_bias", self.genetic_bias);
        dict.set_item("active", self.active);

        dict.into()
    }
}
