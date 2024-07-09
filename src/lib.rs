//! Evo RL is a machine learning library built on the concept of Neuroevolution -- evolving an
//! architecture for neural networks as opposed to pre-specifying it. This library is best suited
//! for Reinforcement Learning tasks in which a reward (or fitness) score can be assigned to
//! agents.
//! 
//! Evo RL is a WIP and is in the pre-alpha state. 

pub mod neuron;
pub mod enecode;
pub mod graph;
pub mod doctest;
pub mod population;
pub mod api;
pub mod agent_wrapper;

use rand::prelude::*;
use enecode::topology::TopologyGene;
use enecode::NeuronType;
use log::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::f32::consts::E;

///Utility function for logging unit tests
pub fn setup_logger() {
    pretty_env_logger::try_init().ok();
}


/// Returns an rng based on Option<seed>
pub fn rng_box(rng_seed: Option<u8>) -> Box<dyn RngCore> {
        match rng_seed {
            Some(seedu8) => {
                let seed = [seedu8; 32];
                Box::new(StdRng::from_seed(seed))
            }
            None => Box::new(rand::thread_rng())
        }
}

/// Convenience function to easily create maps when specifying gene strings. 
pub fn hash_em(names: Vec<&str>, weights: Vec<f32>) -> HashMap<String, f32> {
    let mut hm: HashMap<String, f32> = HashMap::new();
    for (inn_number, weight) in names.iter().zip(weights.iter()) {
        hm.insert(String::from(*inn_number), *weight);
    }

    hm
}

/// Convenience function to easily create a vector of owned Strings. 
pub fn ez_input(names: Vec<&str>) -> Vec<String> {
    names.iter().map(|&n| String::from(n)).collect()
}

/// Sorts topology gene in the order of Input, Hidden, Output and then by innovation number
pub fn sort_genes_by_neuron_type(input_topology_vector: Vec<TopologyGene>) -> Vec<TopologyGene> {
        let mut topology_s: Vec<TopologyGene> = Vec::new();
        let mut topology_hidden: Vec<TopologyGene> = Vec::new();
        let mut topology_outputs: Vec<TopologyGene> = Vec::new();

        for tg in input_topology_vector.into_iter() {

            match tg.pin {
                NeuronType::In => topology_s.push(tg),
                NeuronType::Hidden => topology_hidden.push(tg),
                NeuronType::Out => topology_outputs.push(tg),
            };

        }

        topology_s.sort_by_key(|tg| tg.innovation_number.clone() );
        topology_hidden.sort_by_key(|tg| tg.innovation_number.clone() );
        topology_outputs.sort_by_key(|tg| tg.innovation_number.clone() );

        topology_s.extend(topology_hidden.drain(..));
        topology_s.extend(topology_outputs.drain(..));
        topology_s
    
}

/// Returns progenitor code base for an innnovation number
pub fn progenitor_code(innovation_number: &str) -> &str {
    match innovation_number.find("-") {
        Some(idx) => {
            let(prog, _tail) = innovation_number.split_at(idx);
            prog
        }
        None => innovation_number
    }

}

/// Increments the ID of a neuron when creating a daughter
pub fn increment_innovation_number(neuron_id: &str, daughter_ids: Vec<&str>) -> Arc<str> {
    //innovation numbers will be of the form alphanumeric string (progenitor code) followed by
    //numeric (lineage code)
    //First, identify the progenitor code
    
    let progenitor_code: &str = progenitor_code(neuron_id);

    let daughter_ids_progenitor: Vec<&str> = daughter_ids.iter().map(|x| *x)
                                                                .filter(|id| id.starts_with(progenitor_code))
                                                                .filter(|&id| id != progenitor_code).collect();

    //If it is the first daughter, add -1 to the end of the string
    if daughter_ids_progenitor.len() == 0 {
        format!("{}-0001", progenitor_code).into()
    } else {
        //else increment the largest daughter
        let largest_daughter_id = daughter_ids_progenitor.iter().max().unwrap();

        if let Some(idx) = largest_daughter_id.rfind("-") {
            let (previous_lineage, largest_daughter_number) = largest_daughter_id.split_at(idx);

            let ldn: i32 = match largest_daughter_number[1..].parse() {
                Ok(n) => n,
                Err(_e) => panic!("Failed to parse string daughter number"),
            };

            let mut daughter_id = String::from(previous_lineage);
            //daughter_id.push('-');

            let daughter_innovation = format!("-{:0>4}", ldn + 1);
            daughter_id.push_str(&daughter_innovation);

            Arc::from(daughter_id)

        } else {
            debug!("Problem with parsing string largest_daughter_id {} while duplicating {}", largest_daughter_id, neuron_id);
            panic!("Attempted to parse daughter innovation number but found invalid code");
        }
    }
}

//Non-linearity functions. Thank yourAkshay Ballal for sigmoid and relu
pub fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

//Non-linearity functions. Thank yourAkshay Ballal for sigmoid and relu
pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progenitor_code() {
        assert_eq!("a0", progenitor_code("a0"));
        assert_eq!("a0", progenitor_code("a0-12345"));
        assert_eq!("b00", progenitor_code("b00-12345"));
    }

    #[test]
    fn test_increment_innovation_number() {
        let innovation_number = String::from("a0");
        let daughters = Vec::new();

        let d1 = increment_innovation_number(&innovation_number, daughters);
        assert_eq!(&*d1, "a0-0001");

        let daughters2 = vec!["a0-0001", "a0-0002"];
        let d2 = increment_innovation_number(&innovation_number, daughters2);
        assert_eq!(&*d2, "a0-0003");

        let innovation_number2 = String::from("a0-0001");

        let daughters3 = vec!["a0-0002", "a0-0005", "B0-10000"];
        let d3 = increment_innovation_number(&innovation_number2, daughters3);

        assert_eq!(&*d3, "a0-0006");
    }
}
