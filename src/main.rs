use evo_rl::enecode::{EneCode, TopologyGene, NeuronalPropertiesGene, MetaLearningGene, NeuronType};
use evo_rl::graph::FeedForwardNeuralNetwork;

fn main() {
    //Immediate Goal: Evolve a neural network to solve XOR? 
    //
    //1. Define genetic structur for XOR minimal network, including inputs and outputs
    //2. Test random and init over 1000 turns using backprop
    //3. Move this to a unit Test

    let xor_network_genome = EneCode {
        neuron_id: vec!["i01", "i02", "D", "E"],
        topology: vec![
            TopologyGene {
                innovation_number: "i01",
                pin: NeuronType::In,
                inputs: vec![],
                outputs: vec!["D", "E"],
                genetic_weights: vec![],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "i02",
                pin: NeuronType::In,
                inputs: vec![],
                outputs: vec!["D", "E"],
                genetic_weights: vec![],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "D",
                pin: NeuronType::Hidden,
                inputs: vec!["b", "i01", "i02"],
                outputs: vec!["E"],
                genetic_weights: vec![1., 0.2, 0.3],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "E",
                pin: NeuronType::Out,
                inputs: vec!["b", "i01", "i02", "D"],
                outputs: vec![],
                genetic_weights: vec![1., -0.1, 0.3],
                genetic_bias: 0.,
                active: true,
            },
        ],
        neuronal_props: NeuronalPropertiesGene {
            innovation_number: "p01",
            tau: 0.,
            homeostatic_force: 0.,
            tanh_alpha: 1.,
        },
        meta_learning: MetaLearningGene {
            innovation_number: "m01",
            learning_rate: 0.001,
            learning_threshold: 0.5,
        }

    };

    let mut fnn = FeedForwardNeuralNetwork::new(&xor_network_genome);
    fnn.initialize();

}
