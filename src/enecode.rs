//! Module which specifies the genetic encoding for neural network and neural properties. The
//! genetic encoding is used to reconstruct neural network graphs and for recombination of
//! offspring.

use log::*;
use pyo3::ToPyObject;
use rand::Rng;
use rand::seq::IteratorRandom;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::path::PathBuf;
use std::fs::File;
use std::io::{self, Read};
use thiserror::Error;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Serialize, Deserialize};

use crate::{graph::NeuralNetwork, sort_genes_by_neuron_type};
use crate::enecode::topology::TopologyGene;
use crate::enecode::properties::NeuronalPropertiesGene;
use crate::enecode::meta::MetaLearningGene;

pub mod topology;
pub mod properties;
pub mod meta;

/// `EneCode` encapsulates the genetic blueprint for constructing an entire neural network.
///
/// This struct holds all the information required to instantiate a neural network with
/// distinct neurons, synaptic connections, and meta-learning parameters. It consists of a 
/// collection of genes that provide the blueprint for each individual neuron's topology,
/// its neuronal properties, and meta-learning rules.
///
/// # Fields
/// * `neuron_id` - A vector containing the unique identifiers for each neuron in the genome.
///                 These IDs are used to find the associated topology gene for each neuron.
///
/// * `topology` - A list of `TopologyGene` structs that describe the synaptic connections, 
///                neuron types, and other topological features for each neuron in the network.
///
/// * `neuronal_props` - An instance of `NeuronalPropertiesGene` that provides global neuronal
///                      properties like time constants, homeostatic forces, and activation function
///                      scaling factors.
///
/// * `meta_learning` - An instance of `MetaLearningGene` that provides meta-learning parameters 
///                     such as learning rate and learning thresholds for synaptic adjustments.
///
/// # Example Usage
/// ```rust
/// use evo_rl::enecode::EneCode;
/// use evo_rl::enecode::TopologyGene;
/// use evo_rl::enecode::NeuronalPropertiesGene;
/// use evo_rl::enecode::MetaLearningGene;
/// use evo_rl::enecode::NeuronType;
/// # use std::collections::HashMap;
///
/// // Initialization (example)
/// let genome = EneCode::new_from_genome (
///     vec![
///         TopologyGene {
///             innovation_number: Arc::from("N1"),
///             pin: NeuronType::In,
///             inputs: HashMap::new(),
///             genetic_bias: 0.1,
///             active: true
///         },
///         // ... more TopologyGene
///     ],
///     NeuronalPropertiesGene {
///         innovation_number: Arc::from("NP01"),
///         tau: 0.9,
///         homeostatic_force: 0.1,
///         tanh_alpha: 2.0,
///     },
///     MetaLearningGene {
///         innovation_number: Arc::from("MTL01"),
///         learning_rate: 0.01,
///         learning_threshold: 0.5,
///     });
/// ```

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[pyclass]
pub struct EneCode {
    pub neuron_id: Vec<String>, //equivalent to innovation_number in TopologyGene
    pub topology: Vec<TopologyGene>,
    pub neuronal_props: NeuronalPropertiesGene,
    pub meta_learning: MetaLearningGene,
}

impl ToPyObject for EneCode {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("neuron_id", &self.neuron_id);
        dict.set_item("topology", &self.topology);
        dict.set_item("neuronal_props", &self.neuronal_props);
        dict.set_item("meta_learning", &self.meta_learning);
        
        dict.into()
    }
}

/// Creates genome from neural network after recombination and mutation
impl From<&NeuralNetwork> for EneCode {
    fn from(network: &NeuralNetwork) ->  Self {
        let neuron_id: Vec<String> = network.node_identity_map.keys().map(|id| String::from(id)).collect();

        let mut topology: Vec<TopologyGene> = Vec::new();
        for id in neuron_id.iter() {
            //Identify parent nodes and build HashMap of Weights
            let node = network.node_identity_map[id];
            let node_parents = network.graph.neighbors_directed(node, petgraph::Direction::Incoming);

            let mut input_map: HashMap<String, f32> = HashMap::new();

            for parent in node_parents {
                let parent_id = network.graph[parent].id.clone();
                let edge_id = network.graph.find_edge(parent, node);

                let edge_weight: f32 = match edge_id {
                    Some(w) => *network.graph.edge_weight(w).unwrap(),
                    None => panic!("Edge ID was not found"),
                };

                input_map.insert(String::from(&*parent_id), edge_weight);
            }

            topology.push (
                TopologyGene {
                    innovation_number: network.graph[node].id.clone(),
                    inputs: input_map,
                    pin: network.graph[node].neuron_type.clone(),
                    genetic_bias: network.graph[node].bias,
                    active: true,
                }
            )

        }

        let neuronal_props = network.genome.neuronal_props.clone();
        let meta_learning = network.genome.meta_learning.clone();

        EneCode::new_from_genome(topology, neuronal_props, meta_learning) 
    }
}

impl TryFrom<&PathBuf> for EneCode {

    type Error = serde_json::Error;

    fn try_from(checkpoint: &PathBuf) -> Result<Self, Self::Error> {
        let mut file = File::open(checkpoint).unwrap();
        let mut content = String::new();

        let enecode = match file.read_to_string(&mut content) {
            Ok(_) => serde_json::from_str(&content),
            Err(err) => panic!("{}", err)
        };

        enecode
    }
}

impl EneCode {
    //TODO: Constructor for Vec<ToplogyGene> that has parameters for number of inputs/outputs
    //TODO: Default constructors for all *gene structs
    
    ///Constructor function for basic genome with defined number of inputs/outputs
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {

        // Generate topology with a default rule for hidden progenitors. 
        let topology_s: Vec<TopologyGene> = EneCode::generate_new_topology(num_inputs, num_inputs, num_outputs);
        
        // generate owned string of neuron id
        let neuron_id: Vec<String> = topology_s.iter().map(|x| String::from(&*x.innovation_number)).collect();
        
        Self {
            neuron_id,
            topology: topology_s,
            neuronal_props: NeuronalPropertiesGene::default(),
            meta_learning: MetaLearningGene::default()
        }

    }


    ///Generates an appropriate Vec<TopologyGene> for a given number of inputs and outputs and
    ///hidden neurons. Default is a fully connected FF layer. 
    pub fn generate_new_topology(input_size: usize, output_size: usize, num_hidden: usize) -> Vec<TopologyGene> {

        let mut topology: Vec<TopologyGene> = Vec::new();

        for input_idx in 0..input_size {
            let neuron_id  =  format!("i{:02}", input_idx);

            debug!("adding neuron {}", neuron_id);

            topology.push (
                TopologyGene {
                    innovation_number: Arc::from(neuron_id),
                    pin: NeuronType::In,
                    inputs: HashMap::new(),
                    genetic_bias: 0.,
                    active: true,
                }
            )
        }

        for hidden_idx in 0..num_hidden {
            let mut inputs: Vec<String> = (0..input_size).map(|x| format!("i{:02}", x)).collect();


            let mut input_weight_map: HashMap<String, f32> = HashMap::new();

            for input_id in inputs.drain(..) {
                input_weight_map.insert(input_id, 0.);
            }

            let neuron_id = format!("h{:02}", hidden_idx); 
            debug!("adding neuron {}", neuron_id);

            topology.push (
                TopologyGene {
                    innovation_number: Arc::from(neuron_id),
                    pin: NeuronType::Hidden,
                    inputs: input_weight_map,
                    genetic_bias: 0.,
                    active: true,
                }
            )
        }

        for output_idx in 0..output_size {
            let mut hidden: Vec<String> = (0..num_hidden).map(|x| format!("h{:02}", x)).collect();

            let mut input_weight_map: HashMap<String, f32> = HashMap::new();

            for input_id in hidden.drain(..) {
                input_weight_map.insert(input_id, 0.);
            }

            let neuron_id =  format!("x{:02}", output_idx);
            debug!("adding neuron {}", neuron_id);

            topology.push (
                TopologyGene {
                    innovation_number: Arc::from(neuron_id),
                    pin: NeuronType::Out,
                    inputs: input_weight_map,
                    genetic_bias: 0.,
                    active: true,
                }
            )
        }


        topology
    }

    ///Constructor function for EneCode based on established genome, puts things into correct order based on NeuronType and
    ///innovation number
    pub fn new_from_genome(topology: Vec<TopologyGene>, neuronal_props: NeuronalPropertiesGene, meta_learning: MetaLearningGene) -> Self {

        let topology_s: Vec<TopologyGene> = sort_genes_by_neuron_type(topology);
        let neuron_id: Vec<String> = topology_s.iter().map(|tg| tg.innovation_number.to_string()).collect();

        EneCode {
            neuron_id,
            topology: topology_s,
            neuronal_props,
            meta_learning
        }


    }

    /// Fetches the topology gene corresponding to a given neuron ID.
    ///
    /// # Arguments
    /// * `neuron_id` - The unique identifier for the neuron.
    ///
    /// # Returns
    /// A reference to the `TopologyGene` associated with the neuron ID.
    pub fn topology_gene(&self, neuron_id: &str) -> &TopologyGene {
        let gene = self.topology.iter()
            .find(|&g| &*g.innovation_number == neuron_id)
            .expect("Innovation Number Not Found in Topology Genome!");

        gene
    }

    /// Performs genetic recombination during mating
    ///
    /// # Arguments
    /// * `rng` - thread_rng 
    /// * `other_genome` - the enecode of the partner to recombine with
    ///
    /// # Returns
    /// A Result<EneCode, RecombinationError>
    pub fn recombine<R: Rng>(&self, rng: &mut R, other_genome: &EneCode) -> Result<EneCode, RecombinationError> {
        // determine the number of crossover points by seeing how many genes have matching
        // innovation number
        let self_innovation_nums: HashSet<&str> = self.neuron_id.iter().map(|x| &x[..]).collect();
        let other_innovation_nums: HashSet<&str> = other_genome.neuron_id.iter().map(|x| &x[..]).collect();
        let homology_genes: Vec<&str> = self_innovation_nums.intersection(&other_innovation_nums).map(|x| *x).collect();

        let crossover_topology_vec: Vec<TopologyGene> = self.topology.iter().filter(|x| homology_genes.contains(&&*x.innovation_number))
                                                             .map(|tg| tg.clone()).collect();

        let sorted_crosover_topology: Vec<_> = sort_genes_by_neuron_type(crossover_topology_vec);
        let sorted_homology_genes: Vec<&str> = sorted_crosover_topology.iter().map(|tg| &*tg.innovation_number).collect();

        // determine number of crossover points or return a recombination error if none exists
        if sorted_homology_genes.len() == 0 {
            return Err(RecombinationError::CrossoverMatchError);
        } 

        let max_crossover_points = if sorted_homology_genes.len() == 1 {
            1
        } else { 
            sorted_homology_genes.len() / 2
        };

        let n_crossover = if max_crossover_points < 2 { 1 } else {rng.gen_range(1..=max_crossover_points) };

        // determine location of each crossover point
        let mut crossover_points: Vec<usize> = (0..sorted_homology_genes.len() - 1).choose_multiple(rng, n_crossover);
        crossover_points.sort();
        debug!("Crossover points {:#?}", crossover_points);

        let mut recombined_offspring_topology: Vec<TopologyGene> = Vec::new();

        // for each crossover, swap at the matching innovation number

        //Clone each genome and reverse it for popping
        let mut own_copy: Vec<TopologyGene> = self.topology.clone();
        own_copy.reverse();

        let mut others_copy: Vec<TopologyGene> = other_genome.topology.clone();
        others_copy.reverse();

        let mut use_self = true;

        for point in crossover_points {
            let innovation_number = sorted_homology_genes[point];

            let mut self_genes: Vec<TopologyGene> = Vec::new();
            while let Some(sg) = own_copy.pop() {
                if &*sg.innovation_number == innovation_number {
                    self_genes.push(sg);
                    break;
                }
                self_genes.push(sg);
            }

            let mut other_genes: Vec<TopologyGene> = Vec::new();
            while let Some(og) = others_copy.pop() {
                if &*og.innovation_number == innovation_number {
                    other_genes.push(og);
                    break;
                }
                other_genes.push(og);
            }

            //If innovation number isn't found then there is no corresponding crossover point
            if others_copy.is_empty() {
                return Err(RecombinationError::CrossoverMatchError);
            }

            if use_self {
                recombined_offspring_topology.extend(self_genes.drain(..));
            } else {
                recombined_offspring_topology.extend(other_genes.drain(..));
            }

            use_self = !use_self;

        }

        //Add any remaining genes left over from the last recombination point onwards
        if use_self {
            recombined_offspring_topology.extend(own_copy.drain(..));
        } else {
            recombined_offspring_topology.extend(others_copy.drain(..));
        }

        Ok(EneCode::new_from_genome(recombined_offspring_topology, self.neuronal_props.clone(), self.meta_learning.clone()))
    }


}

/// `NeuronalEneCode` is a struct that encapsulates all genetic information for a single neuron.
///
/// This struct combines the topological, neuronal properties, and meta-learning genes 
/// for an individual neuron in a neural network. It is used to instantiate a 
/// single `Nn` (neuron) in the network graph.
///
/// # Fields
/// * `neuron_id` - The unique identifier for the neuron.
/// * `topology` - The topological gene, which includes information like innovation number, 
///                inputs, outputs, weights, bias, and activation status.
/// * `properties` - The neuronal properties gene, which includes time constants and other 
///                  neuron-specific parameters.
/// * `meta` - The meta-learning gene, which includes learning rates and thresholds.
///
/// # Example
/// ```rust
/// # use evo_rl::doctest::GENOME_EXAMPLE;
/// use evo_rl::enecode::NeuronalEneCode;
/// use evo_rl::enecode::EneCode;
///
/// // Assume `genome` is a properly initialized EneCode
/// # let genome = GENOME_EXAMPLE.clone();
/// let neuron_id = Arc::from("N1");
/// let neuronal_ene_code = NeuronalEneCode::new_from_genome_from_enecode(neuron_id, &genome);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronalEneCode<'a> {
    pub neuron_id: Arc<str>,
    pub topology: &'a TopologyGene,
    pub properties: &'a NeuronalPropertiesGene,
    pub meta: &'a MetaLearningGene,
}

/// Generates a more specific genetic handle for use in initializing a neuron
impl<'a> NeuronalEneCode<'a> {
    pub fn new_from_enecode(neuron_id: &str, genome: &'a EneCode) -> Self {
        let topology = genome.topology_gene(&neuron_id);
        NeuronalEneCode {
            neuron_id: neuron_id.into(),
            topology,
            properties: &genome.neuronal_props,
            meta: &genome.meta_learning, 
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NeuronType {
    In,
    Out,
    Hidden,
}

#[derive(Debug, Error)]
pub enum RecombinationError {
    #[error("Incompatible Genomes: Non-matching innovation number chosen for crossover!")]
    CrossoverMatchError
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use assert_matches::assert_matches;
    use crate::doctest::{XOR_GENOME, GENOME_EXAMPLE, GENOME_EXAMPLE2, XOR_GENOME_MINIMAL};
    use crate::setup_logger;

    #[test] 
    fn test_generate_new_topology() {
        setup_logger();

        let topology = EneCode::generate_new_topology(1, 1, 1);
        assert_eq!(topology.len(), 3);

        let h00: &TopologyGene = topology.iter().filter(|&t| &*t.innovation_number == "h00").next().unwrap(); 
        let h00_input_keys: Vec<&str> = h00.inputs.iter().map(|(x, _)| x.as_str()).collect();
        assert_eq!(vec!["i00"], h00_input_keys);

        let x00: &TopologyGene = topology.iter().filter(|&t| &*t.innovation_number == "x00").next().unwrap(); 
        let x00_input_keys: Vec<&str> = x00.inputs.iter().map(|(x, _)| x.as_str()).collect();
        assert_eq!(vec!["h00"], x00_input_keys);

        let topology2 = EneCode::generate_new_topology(2, 2, 2);
        assert_eq!(topology2.len(), 6);

        let h01: &TopologyGene = topology2.iter().filter(|&t| &*t.innovation_number == "h01").next().unwrap(); 
        let mut h01_input_keys: Vec<&str> = h01.inputs.iter().map(|(x, _)| x.as_str()).collect();
        h01_input_keys.sort();

        assert_eq!(vec!["i00", "i01"], h01_input_keys);

        let x01: &TopologyGene = topology2.iter().filter(|&t| &*t.innovation_number == "x01").next().unwrap(); 
        let mut x01_input_keys: Vec<&str> = x01.inputs.iter().map(|(x, _)| x.as_str()).collect();
        x01_input_keys.sort();

        assert_eq!(vec!["h00", "h01"], x01_input_keys);

    }

    #[test]
    fn test_new_from_enecode() {
        // Create an EneCode instance and use it to initialize a NeuronalEneCode

        let neuronal_ene_code = NeuronalEneCode::new_from_enecode("N1", &GENOME_EXAMPLE);

        let mut input_map = HashMap::new();
        input_map.insert(String::from("input_1"), 1.0_f32);

        let expected_nec: NeuronalEneCode = NeuronalEneCode {
         neuron_id: Arc::from("N1"),
         topology: &TopologyGene {
                 innovation_number: Arc::from("N1"),
                 pin: NeuronType::Hidden,
                 inputs: input_map,
                 genetic_bias: 0.0,
                 active: true }, 
        properties: &GENOME_EXAMPLE.neuronal_props,
        meta: &GENOME_EXAMPLE.meta_learning,
        };

        // Validate that the properties have been copied over correctly
        assert_eq!(neuronal_ene_code, expected_nec);
    }

    #[test]
    fn test_enecode_from_neural_network() {
        let genome = GENOME_EXAMPLE.clone();
        let genome_comparison = GENOME_EXAMPLE.clone();
        let network_example = NeuralNetwork::new(genome);

        let test_enecode = EneCode::from(&network_example);

        assert_eq!(test_enecode, genome_comparison);
    }

    #[test]
    fn test_topology_gene() {
        let topology_gene_n1 = GENOME_EXAMPLE.topology_gene(&String::from("N1"));
        assert_eq!(String::from("N1"), topology_gene_n1.innovation_number.to_string());
    }

    #[test]
    fn test_recombine_same_everything_short_genome() {
        let seed = [0; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let ene1 = GENOME_EXAMPLE.clone();
        let ene2 = GENOME_EXAMPLE.clone();

        let recombined = ene1.recombine(&mut rng, &ene2).unwrap();

        assert_eq!(recombined.neuron_id.len(), ene1.neuron_id.len());
    }

    #[test]
    fn test_recombine_same_everything_long_genome() {
        let seed = [0; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let ene1 = XOR_GENOME.clone();
        let ene2 = XOR_GENOME.clone();

        let recombined = ene1.recombine(&mut rng, &ene2).unwrap();

        assert_eq!(recombined.neuron_id.len(), ene1.neuron_id.len());
    }

    #[test]
    fn test_recombine_missing_gene_long_genome() {
        let seed = [0; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let ene1 = XOR_GENOME.clone();
        let ene2 = XOR_GENOME_MINIMAL.clone();

        let recombined = ene1.recombine(&mut rng, &ene2).unwrap();
        let crossover_genes: Vec<&String> = recombined.neuron_id.iter().filter(|&id| id == "A").collect();

        assert!(crossover_genes.len() == 1);
        assert_eq!(crossover_genes[0], "A");
    }


    #[test]
    fn test_recombine_same_topology_different_genetic_bias() {
        setup_logger();

        let seed = [17; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let ene1 = XOR_GENOME.clone();
        let ene2_base = XOR_GENOME.clone();

        let new_topology_genome: Vec<TopologyGene> = ene2_base.topology.iter().map( |tg| 
                TopologyGene {
                    innovation_number: tg.innovation_number.clone(),
                    pin: tg.pin.clone(),
                    inputs: tg.inputs.clone(),
                    genetic_bias: 5.,
                    active: tg.active
                }
            ).collect();

        let ene2: EneCode = EneCode::new_from_genome(new_topology_genome, ene2_base.neuronal_props, ene2_base.meta_learning);

        let recombined = ene1.recombine(&mut rng, &ene2).unwrap();
        let recombined_genetic_bias: Vec<_> = recombined.topology.iter().map(|tg| tg.genetic_bias).collect();

        info!("Recombined bias vector {:#?}", recombined_genetic_bias);
        assert_ne!(recombined_genetic_bias, vec![0., 0., 0., 0.]);
        assert_ne!(recombined_genetic_bias, vec![5., 5., 5., 5.]);
        assert_eq!(recombined_genetic_bias.len(), ene1.neuron_id.len());

    }

    #[test]
    fn test_recombine_different_topology_compatible_genomes() {
        let seed = [0; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let ene1 = GENOME_EXAMPLE.clone();
        let ene2 = GENOME_EXAMPLE2.clone();

        let recombined = ene1.recombine(&mut rng, &ene2).unwrap();

        assert!(recombined.neuron_id.len() == 4);
    }


    #[test]
    fn test_recombine_incompatible_genomes() {
        let seed = [0; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let ene1 = GENOME_EXAMPLE.clone();
        let ene2 = XOR_GENOME.clone();

        let recombined = ene1.recombine(&mut rng, &ene2);

        assert_matches!(recombined, Err(RecombinationError::CrossoverMatchError));
    }
}

