use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::ToPyObject;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use serde::de::{self, Deserializer, Visitor, MapAccess};
use serde::Deserialize;
use std::collections::HashMap;
use pyo3::types::PyDict;
use crate::enecode::NeuronType;
use std::fmt;

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
    pub inputs: HashMap<String, f64>, //map that defines genetic weight of synapse for each parent
    pub genetic_bias: f64,
    pub active: bool,
}

impl Default for TopologyGene {
    fn default() -> Self {
        let inputs: HashMap<String, f64> = HashMap::new();
        Self {
            innovation_number: Arc::from("n01"),
            pin: NeuronType::Hidden,
            inputs,
            genetic_bias: 0.0,
            active: true,
        }
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

impl<'de> Deserialize<'de> for TopologyGene {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { Innovation_Number, Pin, Inputs, Genetic_Bias, Active }

        struct TopologyGeneVisitor;

        impl<'de> Visitor<'de> for TopologyGeneVisitor {
            type Value = TopologyGene;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TopologyGene")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TopologyGene, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut innovation_number = None;
                let mut pin = None;
                let mut inputs = None;
                let mut genetic_bias = None;
                let mut active = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Innovation_Number => {
                            if innovation_number.is_some() {
                                return Err(de::Error::duplicate_field("innovation_number"));
                            }
                            let value: String = map.next_value()?;
                            innovation_number = Some(Arc::from(value.as_str()));
                        }
                        Field::Pin => {
                            if pin.is_some() {
                                return Err(de::Error::duplicate_field("pin"));
                            }
                            pin = Some(map.next_value()?);
                        }
                        Field::Inputs => {
                            if inputs.is_some() {
                                return Err(de::Error::duplicate_field("inputs"));
                            }
                            inputs = Some(map.next_value()?);
                        }
                        Field::Genetic_Bias => {
                            if genetic_bias.is_some() {
                                return Err(de::Error::duplicate_field("genetic_bias"));
                            }
                            genetic_bias = Some(map.next_value()?);
                        }
                        Field::Active => {
                            if active.is_some() {
                                return Err(de::Error::duplicate_field("active"));
                            }
                            active = Some(map.next_value()?);
                        }
                    }
                }

                let innovation_number = innovation_number.ok_or_else(|| de::Error::missing_field("innovation_number"))?;
                let pin = pin.ok_or_else(|| de::Error::missing_field("pin"))?;
                let inputs = inputs.ok_or_else(|| de::Error::missing_field("inputs"))?;
                let genetic_bias = genetic_bias.ok_or_else(|| de::Error::missing_field("genetic_bias"))?;
                let active = active.ok_or_else(|| de::Error::missing_field("active"))?;

                Ok(TopologyGene {
                    innovation_number,
                    pin,
                    inputs,
                    genetic_bias,
                    active
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["innovation_number", "pin", "inputs", "genetic_bias", "active"];
        deserializer.deserialize_struct("TopologyGene", FIELDS, TopologyGeneVisitor)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use log::*;
    use crate::setup_logger;

    #[test]
    fn test_serialize_topology() {
        setup_logger();

        let tpg: TopologyGene = TopologyGene::default();
        let json = serde_json::to_string_pretty(&tpg).unwrap();
        debug!("{}", json);

        assert!(json.len() > 0);
    }

    #[test]
    fn test_deserialize_topology() {
        setup_logger();

        let tpg: TopologyGene = TopologyGene::default();
        let json = serde_json::to_string_pretty(&tpg).unwrap();

        let tpg_deserialized: TopologyGene = serde_json::from_str(&json).unwrap();

        assert_eq!(TopologyGene::default(), tpg_deserialized);

    }

}
