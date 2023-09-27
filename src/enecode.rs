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
/// use crate::enecode::EneCode;
/// use crate::enecode::TopologyGene;
/// use crate::enecode::NeuronalPropertiesGene;
/// use crate::enecode::MetaLearningGene;
/// use crate::enecode::NeuronType;
///
/// // Initialization (example)
/// let genome = EneCode {
///     neuron_id: vec!["N1".to_string(), "N2".to_string()],
///     topology: vec![
///         TopologyGene {
///             innovation_number: "N1".to_string(),
///             pin: NeuronType::In,
///             inputs: vec![],
///             outputs: vec!["N2".to_string()],
///             genetic_weights: vec![0.5],
///             genetic_bias: 0.1,
///             active: true
///         },
///         // ... more TopologyGene
///     ],
///     neuronal_props: NeuronalPropertiesGene {
///         innovation_number: "Global".to_string(),
///         tau: 0.9,
///         homeostatic_force: 0.1,
///         tanh_alpha: 2.0,
///     },
///     meta_learning: MetaLearningGene {
///         innovation_number: "Global".to_string(),
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
/// use crate::enecode::NeuronalEneCode;
/// use crate::enecode::EneCode;
///
/// // Assume `genome` is a properly initialized EneCode
/// let neuron_id = "some_id";
/// let neuronal_ene_code = NeuronalEneCode::new_from_enecode(neuron_id, &genome);
/// ```
pub struct NeuronalEneCode {
    pub neuron_id: String,
    pub topology: TopologyGene,
    pub properties: NeuronalPropertiesGene,
    pub meta: MetaLearningGene,
}

impl NeuronalEneCode {
    pub fn new_from_enecode(neuron_id: String, genome: &EneCode) -> Self {
        let topology = genome.topology_gene(&neuron_id).clone();
        NeuronalEneCode {
            neuron_id,
            topology,
            properties: genome.neuronal_props.clone(),
            meta: genome.meta_learning.clone(), 
        }
    }
}

#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct TopologyGene {
    pub innovation_number: String,
    pub pin: NeuronType, //stolen from python-neat for outside connections
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub genetic_weights: Vec<f32>,
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct MetaLearningGene {
    pub innovation_number: String,
    pub learning_rate: f32,
    pub learning_threshold: f32
}
