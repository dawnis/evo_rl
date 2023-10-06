use rand::Rng;
use rand::seq::IteratorRandom;
use std::collections::HashMap;

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
/// let genome = EneCode {
///     neuron_id: vec!["N1".to_string(), "N2".to_string()],
///     topology: vec![
///         TopologyGene {
///             innovation_number: "N1".to_string(),
///             pin: NeuronType::In,
///             inputs: HashMap::new(),
///             outputs: vec!["N2".to_string()],
///             genetic_bias: 0.1,
///             active: true
///         },
///         // ... more TopologyGene
///     ],
///     neuronal_props: NeuronalPropertiesGene {
///         innovation_number: "NP01".to_string(),
///         tau: 0.9,
///         homeostatic_force: 0.1,
///         tanh_alpha: 2.0,
///     },
///     meta_learning: MetaLearningGene {
///         innovation_number: "MTL01".to_string(),
///         learning_rate: 0.01,
///         learning_threshold: 0.5,
///     },
/// };
/// ```
#[derive(Debug, Clone)]
pub struct EneCode {
    pub neuron_id: Vec<String>, //equivalent to innovation_number in TopologyGene
    pub topology: Vec<TopologyGene>,
    pub neuronal_props: NeuronalPropertiesGene,
    pub meta_learning: MetaLearningGene,
}

impl EneCode {
    /// Fetches the topology gene corresponding to a given neuron ID.
    ///
    /// # Arguments
    /// * `neuron_id` - The unique identifier for the neuron.
    ///
    /// # Returns
    /// A reference to the `TopologyGene` associated with the neuron ID.
    pub fn topology_gene(&self, neuron_id: &String) -> &TopologyGene {
        let gene = self.topology.iter()
            .find(|&g| g.innovation_number == *neuron_id)
            .expect("Innovation Number Not Found in Topology Genome!");

        gene
    }

    pub fn recombine(&self, other_genome: &EneCode) -> EneCode {
        // first, determine number of crossover points
        let mut rng = rand::thread_rng();
        let max_crossover_points = self.neuron_id.len() / 2;
        let n_crossover = rng.gen_range(1..max_crossover_points);

        // second determine location of each crossover point
        let mut crossover_points: Vec<usize> = (0..self.neuron_id.len()).choose_multiple(&mut rng, n_crossover);
        crossover_points.sort();

        let mut recombined_offspring_topology: Vec<TopologyGene> = Vec::new();

        // for each crossover, swap at the matching innovation number

        //Clone each genome and reverse it for popping
        let mut own_copy: Vec<TopologyGene> = self.topology.clone();
        own_copy.reverse();

        let mut others_copy: Vec<TopologyGene> = other_genome.topology.clone();
        others_copy.reverse();

        let use_self = true;

        for point in crossover_points {
            let innovation_number = &self.neuron_id[point];

            let mut self_genes: Vec<TopologyGene> = Vec::new();
            while let Some(sg) = own_copy.pop() {
                if sg.innovation_number == *innovation_number {
                    break;
                }
                self_genes.push(sg);
            }

            let mut other_genes: Vec<TopologyGene> = Vec::new();
            while let Some(og) = others_copy.pop() {
                if og.innovation_number == *innovation_number {
                    break;
                }
                other_genes.push(og);
            }

            if use_self {
                recombined_offspring_topology.extend(self_genes.drain(..));
            } else {
                recombined_offspring_topology.extend(others_copy.drain(..));
            }

        }


        EneCode {
            neuron_id: recombined_offspring_topology.iter().map(|tg| String::from(&tg.innovation_number)).collect(),
            topology: recombined_offspring_topology,
            neuronal_props: self.neuronal_props.clone(),
            meta_learning: self.meta_learning.clone(),
        }
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
/// let neuron_id = "N1".to_string();
/// let neuronal_ene_code = NeuronalEneCode::new_from_enecode(neuron_id, &genome);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronalEneCode<'a> {
    pub neuron_id: String,
    pub topology: &'a TopologyGene,
    pub properties: &'a NeuronalPropertiesGene,
    pub meta: &'a MetaLearningGene,
}

impl<'a> NeuronalEneCode<'a> {
    pub fn new_from_enecode(neuron_id: String, genome: &'a EneCode) -> Self {
        let topology = genome.topology_gene(&neuron_id);
        NeuronalEneCode {
            neuron_id,
            topology,
            properties: &genome.neuronal_props,
            meta: &genome.meta_learning, 
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NeuronType {
    In,
    Out,
    Hidden,
}

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
    pub innovation_number: String,
    pub pin: NeuronType, //stolen from python-neat for outside connections
    pub inputs: HashMap<String, f32>, //map that defines genetic weight of synapse for each parent
    pub outputs: Vec<String>,
    pub genetic_bias: f32,
    pub active: bool,
}

/// Gene that defines the neuronal properties.
///
/// # Fields
/// * `innovation_number` - Unique identifier for this particular gene.
/// * `tau` - The time constant for the neuron.
/// * `homeostatic_force` - Homeostatic force for neuron.
/// * `tanh_alpha` - Scaling factor for tanh activation function.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronalPropertiesGene {
    pub innovation_number: String,
    pub tau: f32,
    pub homeostatic_force: f32,
    pub tanh_alpha: f32,
}

/// Gene that defines the meta-learning rules for the neural network.
///
/// # Fields
/// * `innovation_number` - Unique identifier for this particular gene.
/// * `learning_rate` - Learning rate for synaptic adjustments.
/// * `learning_threshold` - Learning threshold for synaptic adjustments.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaLearningGene {
    pub innovation_number: String,
    pub learning_rate: f32,
    pub learning_threshold: f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctest::GENOME_EXAMPLE;

    #[test]
    fn test_new_from_enecode() {
        // Create an EneCode instance and use it to initialize a NeuronalEneCode

        let neuronal_ene_code = NeuronalEneCode::new_from_enecode(String::from("N1"), &GENOME_EXAMPLE);

        let mut input_map = HashMap::new();
        input_map.insert(String::from("input_1"), 1.0_f32);

        let expected_nec: NeuronalEneCode = NeuronalEneCode {
         neuron_id: String::from("N1"),
         topology: &TopologyGene {
                 innovation_number: "N1".to_string(),
                 pin: NeuronType::Hidden,
                 inputs: input_map,
                 outputs: vec!["output_1".to_string()],
                 genetic_bias: 0.0,
                 active: true }, 
        properties: &GENOME_EXAMPLE.neuronal_props,
        meta: &GENOME_EXAMPLE.meta_learning,
        };

        // Validate that the properties have been copied over correctly
        assert_eq!(neuronal_ene_code, expected_nec);
    }

    #[test]
    fn test_topology_gene() {
        let topology_gene_n1 = GENOME_EXAMPLE.topology_gene(&String::from("N1"));
        assert_eq!(String::from("N1"), topology_gene_n1.innovation_number);
    }

    #[test]
    fn test_recombine() {
        let ene1 = GENOME_EXAMPLE.clone();
        let ene2 = GENOME_EXAMPLE.clone();

        let recombined = ene1.recombine(&ene2);

        assert_eq!(recombined.neuron_id.len(), ene1.neuron_id.len());
    }
}

