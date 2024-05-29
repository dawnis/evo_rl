use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::ToPyObject;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use serde::de::{self, Deserializer, Visitor, MapAccess};
use serde::Deserialize;
use pyo3::types::PyDict;
use std::fmt;
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


impl<'de> Deserialize<'de> for MetaLearningGene {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { InnovationNumber, LearningRate, LearningThreshold }

        struct MetaLearningGeneVisitor;

        impl<'de> Visitor<'de> for MetaLearningGeneVisitor {
            type Value = MetaLearningGene;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct MetaLearningGene")
            }

            fn visit_map<V>(self, mut map: V) -> Result<MetaLearningGene, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut innovation_number = None;
                let mut learning_rate = None;
                let mut learning_threshold = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::InnovationNumber => {
                            if innovation_number.is_some() {
                                return Err(de::Error::duplicate_field("innovation_number"));
                            }
                            let value: String = map.next_value()?;
                            innovation_number = Some(Arc::from(value.as_str()));
                        }
                        Field::LearningRate => {
                            if learning_rate.is_some() {
                                return Err(de::Error::duplicate_field("learning_rate"));
                            }
                            learning_rate = Some(map.next_value()?);
                        }
                        Field::LearningThreshold => {
                            if learning_threshold.is_some() {
                                return Err(de::Error::duplicate_field("learning_threshold"));
                            }
                            learning_threshold = Some(map.next_value()?);
                        }
                    }
                }

                let innovation_number = innovation_number.ok_or_else(|| de::Error::missing_field("innovation_number"))?;
                let learning_rate = learning_rate.ok_or_else(|| de::Error::missing_field("learning_rate"))?;
                let learning_threshold = learning_threshold.ok_or_else(|| de::Error::missing_field("learning_threshold"))?;

                Ok(MetaLearningGene {
                    innovation_number,
                    learning_rate,
                    learning_threshold,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["innovation_number", "learning_rate", "learning_threshold"];
        deserializer.deserialize_struct("MetaLearningGene", FIELDS, MetaLearningGeneVisitor)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use log::*;
    use crate::setup_logger;

    #[test]
    fn test_serialize_metalearning() {
        setup_logger();

        let mtg: MetaLearningGene = MetaLearningGene::default();
        let json = serde_json::to_string_pretty(&mtg).unwrap();
        debug!("{}", json);

        assert!(json.len() > 0);
    }

}
