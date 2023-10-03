use crate::enecode::*;
use std::collections::HashMap;
use lazy_static::lazy_static;


fn hash_em(names: Vec<&str>, weights: Vec<f32>) -> HashMap<String, f32> {
    let mut hm: HashMap<String, f32> = HashMap::new();
    for (inn_number, weight) in names.iter().zip(weights.iter()) {
        hm.insert(String::from(*inn_number), *weight);
    }

    hm
}

fn ez_input(names: Vec<&str>) -> Vec<String> {
    names.iter().map(|&n| String::from(n)).collect()
}

lazy_static! {
    pub static ref GENOME_EXAMPLE: EneCode =  EneCode {
     neuron_id: ez_input(vec!["input_1", "N1", "output_1"]),
     topology: vec![
         TopologyGene {
             innovation_number: "input_1".to_string(),
             pin: NeuronType::In,
             inputs: HashMap::new(),
             outputs: vec!["N1".to_string()],
             genetic_bias: 0.0,
             active: true
         },
         TopologyGene {
             innovation_number: "N1".to_string(),
             pin: NeuronType::Hidden,
             inputs: hash_em(vec!["input_1"], vec![1.0]),
             outputs: vec!["output_1".to_string()],
             genetic_bias: 0.0,
             active: true
         },
         TopologyGene {
             innovation_number: "output_1".to_string(),
             pin: NeuronType::Out,
             inputs: hash_em(vec!["N1"], vec![0.5]),
             outputs: vec![],
             genetic_bias: 0.0,
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

    pub static ref TOPOLOGY_GENE_EXAMPLE: TopologyGene = TopologyGene {
            innovation_number: "h01".to_string(),
            pin: NeuronType::Hidden,
            inputs: hash_em(vec!["i01"], vec![2.]),
            outputs: vec!["F".to_string()],
            genetic_bias: 5.,
            active: true,
    };

    pub static ref META_GENE_EXAMPLE: MetaLearningGene = MetaLearningGene {
            innovation_number: "mtg01".to_string(),
            learning_rate: 0.001,
            learning_threshold: 0.5,
    };

    pub static ref NEURONAL_PROPERTIES_GENE_EXAMPLE: NeuronalPropertiesGene = NeuronalPropertiesGene {
            innovation_number: "npg01".to_string(),
            tau: 0.,
            homeostatic_force: 0., 
            tanh_alpha: 1.,
    };

    pub static ref XOR_GENOME: EneCode = EneCode {
        neuron_id: ez_input(vec!["i01", "i02", "D", "E"]),
        topology: vec![
            TopologyGene {
                innovation_number: "i01".to_string(),
                pin: NeuronType::In,
                inputs: HashMap::new(),
                outputs: vec!["D".to_string(), "E".to_string()],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "i02".to_string(),
                pin: NeuronType::In,
                inputs: HashMap::new(),
                outputs: vec!["D".to_string(), "E".to_string()],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "D".to_string(),
                pin: NeuronType::Hidden,
                inputs: hash_em(vec!["i01", "i02"], vec![0.2, 0.3]),
                outputs: vec!["E".to_string()],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "E".to_string(),
                pin: NeuronType::Out,
                inputs: hash_em(vec!["i01", "i02", "D"], vec![-0.1, 0.3, 0.4]),
                outputs: vec![],
                genetic_bias: 0.,
                active: true,
            },
        ],
        neuronal_props: NeuronalPropertiesGene {
            innovation_number: "p01".to_string(),
            tau: 0.,
            homeostatic_force: 0.,
            tanh_alpha: 1.,
        },
        meta_learning: MetaLearningGene {
            innovation_number: "m01".to_string(),
            learning_rate: 0.001,
            learning_threshold: 0.5,
        }

    };

}



