use crate::enecode::*;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref GENOME_EXAMPLE: EneCode =  EneCode {
     neuron_id: vec!["input_1".to_string(), "N1".to_string(), "output_1".to_string()],
     topology: vec![
         TopologyGene {
             innovation_number: "input_1".to_string(),
             pin: NeuronType::In,
             inputs: vec![],
             outputs: vec!["N1".to_string()],
             genetic_weights: vec![0.5],
             genetic_bias: 0.1,
             active: true
         },
         TopologyGene {
             innovation_number: "N1".to_string(),
             pin: NeuronType::Hidden,
             inputs: vec!["input_1".to_string()],
             outputs: vec!["output_1".to_string()],
             genetic_weights: vec![0.5],
             genetic_bias: 0.1,
             active: true
         },
         TopologyGene {
             innovation_number: "output_1".to_string(),
             pin: NeuronType::Out,
             inputs: vec!["N1".to_string()],
             outputs: vec![],
             genetic_weights: vec![0.5],
             genetic_bias: 0.1,
             active: true
         },
         // ... more TopologyGene
     ],
     neuronal_props: NeuronalPropertiesGene {
         innovation_number: "NP01".to_string(),
         tau: 0.9,
         homeostatic_force: 0.1,
         tanh_alpha: 2.0,
     },
     meta_learning: MetaLearningGene {
         innovation_number: "MTL01".to_string(),
         learning_rate: 0.01,
         learning_threshold: 0.5,
     },
    };

}



