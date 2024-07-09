//! This module contains constants and examples used in unit tests. 

use crate::enecode::*;
use crate::enecode::meta::MetaLearningGene;
use crate::enecode::topology::TopologyGene;
use crate::enecode::properties::NeuronalPropertiesGene;
use crate::hash_em;
use std::sync::Arc;
use std::collections::HashMap;
use lazy_static::lazy_static;


lazy_static! {
    pub static ref GENOME_EXAMPLE: EneCode =  EneCode::new_from_genome(
     vec![
         TopologyGene {
             innovation_number: Arc::from("input_1"),
             pin: NeuronType::In,
             inputs: HashMap::new(),
             genetic_bias: 0.0,
             active: true
         },
         TopologyGene {
             innovation_number: Arc::from("N1"),
             pin: NeuronType::Hidden,
             inputs: hash_em(vec!["input_1"], vec![1.0]),
             genetic_bias: 0.0,
             active: true
         },
         TopologyGene {
             innovation_number: Arc::from("output_1"),
             pin: NeuronType::Out,
             inputs: hash_em(vec!["N1"], vec![0.5]),
             genetic_bias: 0.0,
             active: true
         },
         // ... more TopologyGene
     ],
     NeuronalPropertiesGene {
         innovation_number: Arc::from("NP01"),
         module: Arc::from("Rocinante"),
         tau: 0.9,
         homeostatic_force: 0.1,
         tanh_alpha: 1.,
     },
     MetaLearningGene {
         innovation_number: Arc::from("MTL01"),
         learning_rate: 0.01,
         learning_threshold: 0.5,
     },
    );

    pub static ref TOPOLOGY_GENE_EXAMPLE: TopologyGene = TopologyGene {
            innovation_number: Arc::from("h01"),
            pin: NeuronType::Hidden,
            inputs: hash_em(vec!["i01"], vec![2.]),
            genetic_bias: 5.,
            active: true,
    };

    pub static ref GENOME_EXAMPLE2: EneCode =  EneCode::new_from_genome (
     vec![
         TopologyGene {
             innovation_number: Arc::from("input_1"),
             pin: NeuronType::In,
             inputs: HashMap::new(),
             genetic_bias: 0.0,
             active: true
         },
         TopologyGene {
             innovation_number: Arc::from("N1"),
             pin: NeuronType::Hidden,
             inputs: hash_em(vec!["input_1"], vec![1.0]),
             genetic_bias: 0.0,
             active: true
         },
         TopologyGene {
             innovation_number: Arc::from("N2"),
             pin: NeuronType::Hidden,
             inputs: hash_em(vec!["input_1", "N1"], vec![1.0, -5.0]),
             genetic_bias: 0.0,
             active: true
         },
         TopologyGene {
             innovation_number: Arc::from("output_1"),
             pin: NeuronType::Out,
             inputs: hash_em(vec!["N1", "N2"], vec![0.5, 0.5]),
             genetic_bias: 0.0,
             active: true
         },
         // ... more TopologyGene
     ],
     NeuronalPropertiesGene {
         innovation_number: Arc::from("NP01"),
         module: Arc::from("Rocinante"),
         tau: 0.9,
         homeostatic_force: 0.1,
         tanh_alpha: 1.,
     },
     MetaLearningGene {
         innovation_number: Arc::from("MTL01"),
         learning_rate: 0.01,
         learning_threshold: 0.5,
     },
    );

    pub static ref META_GENE_EXAMPLE: MetaLearningGene = MetaLearningGene {
            innovation_number: Arc::from("mtg01"),
            learning_rate: 0.001,
            learning_threshold: 0.5,
    };

    pub static ref NEURONAL_PROPERTIES_GENE_EXAMPLE: NeuronalPropertiesGene = NeuronalPropertiesGene {
            innovation_number: Arc::from("npg01"),
            module: Arc::from("Engine"),
            tau: 0.,
            homeostatic_force: 0., 
            tanh_alpha: 1.,
    };

    pub static ref XOR_GENOME_MINIMAL: EneCode = EneCode::new_from_genome (
        vec![
            TopologyGene {
                innovation_number: Arc::from("i01"),
                pin: NeuronType::In,
                inputs: HashMap::new(),
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: Arc::from("i02"),
                pin: NeuronType::In,
                inputs: HashMap::new(),
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: Arc::from("A"),
                pin: NeuronType::Hidden,
                inputs: hash_em(vec!["i01", "i02"], vec![0., 0.]),
                genetic_bias: 0.,
                active: true,
            },
            TopologyGene {
                innovation_number: Arc::from("XOR"),
                pin: NeuronType::Out,
                inputs: hash_em(vec!["A"], vec![0.]),
                genetic_bias: 0.,
                active: true,
            },
        ],
        NeuronalPropertiesGene::default(),
        MetaLearningGene::default()
    );

    pub static ref XOR_GENOME: EneCode = EneCode::new_from_genome (
        vec![
            TopologyGene {
                innovation_number: Arc::from("i01"),
                pin: NeuronType::In,
                inputs: HashMap::new(),
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: Arc::from("i02"),
                pin: NeuronType::In,
                inputs: HashMap::new(),
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: Arc::from("A"),
                pin: NeuronType::Hidden,
                inputs: hash_em(vec!["i01", "i02"], vec![0., 0.]),
                genetic_bias: 0.,
                active: true,
            },
            TopologyGene {
                innovation_number: Arc::from("B"),
                pin: NeuronType::Hidden,
                inputs: hash_em(vec!["i01", "i02"], vec![0., 0.]),
                genetic_bias: 0.,
                active: true,
            },
            TopologyGene {
                innovation_number: Arc::from("XOR"),
                pin: NeuronType::Out,
                inputs: hash_em(vec!["A", "B"], vec![0., 0.]),
                genetic_bias: 0.,
                active: true,
            },
        ],
        NeuronalPropertiesGene::default(),
        MetaLearningGene::default()
    );

}



