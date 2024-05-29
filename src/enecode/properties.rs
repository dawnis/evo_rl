use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::ToPyObject;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use serde::Deserialize;
use serde::de::{self, Deserializer, Visitor, MapAccess};
use pyo3::types::PyDict;
use std::fmt;

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

impl Serialize for NeuronalPropertiesGene {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> 
    where 
        S: Serializer,
        {
            let mut state = serializer.serialize_struct("NeuronalPropertiesGene", 4)?;
            state.serialize_field("innovation_number", &self.innovation_number.as_ref())?;
            state.serialize_field("tau", &self.tau)?;
            state.serialize_field("homeostatic_force", &self.homeostatic_force)?;
            state.serialize_field("tanh_alpha", &self.tanh_alpha)?;
            state.end()
        }
}


impl<'de> Deserialize<'de> for NeuronalPropertiesGene {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { InnovationNumber, Tau, HomeostaticForce, TanhAlpha}

        struct NeuronalPropertiesGeneVisitor;

        impl<'de> Visitor<'de> for NeuronalPropertiesGeneVisitor {
            type Value = NeuronalPropertiesGene;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct NeuronalPropertiesGene")
            }

            fn visit_map<V>(self, mut map: V) -> Result<NeuronalPropertiesGene, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut innovation_number = None;
                let mut tau = None;
                let mut homeostatic_force = None;
                let mut tanh_alpha = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::InnovationNumber => {
                            if innovation_number.is_some() {
                                return Err(de::Error::duplicate_field("innovation_number"));
                            }
                            let value: String = map.next_value()?;
                            innovation_number = Some(Arc::from(value.as_str()));
                        }
                        Field::Tau => {
                            if tau.is_some() {
                                return Err(de::Error::duplicate_field("tau"));
                            }
                            tau = Some(map.next_value()?);
                        }
                        Field::HomeostaticForce => {
                            if homeostatic_force.is_some() {
                                return Err(de::Error::duplicate_field("homeostatic_force"));
                            }
                            homeostatic_force = Some(map.next_value()?);
                        }
                        Field::TanhAlpha => {
                            if tanh_alpha.is_some() {
                                return Err(de::Error::duplicate_field("tanh_alpha"));
                            }
                            tanh_alpha = Some(map.next_value()?);
                        }
                    }
                }

                let innovation_number = innovation_number.ok_or_else(|| de::Error::missing_field("innovation_number"))?;
                let tau = tau.ok_or_else(|| de::Error::missing_field("tau"))?;
                let homeostatic_force = homeostatic_force.ok_or_else(|| de::Error::missing_field("homeostatic_force"))?;
                let tanh_alpha = tanh_alpha.ok_or_else(|| de::Error::missing_field("homeostatic_force"))?;

                Ok(NeuronalPropertiesGene {
                    innovation_number,
                    tau,
                    homeostatic_force,
                    tanh_alpha
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["innovation_number", "tau", "homeostatic_force", "tanh_alpha"];
        deserializer.deserialize_struct("NeuronalPropertiesGene", FIELDS, NeuronalPropertiesGeneVisitor)
    }
}

