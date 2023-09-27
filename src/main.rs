use evo_rl::enecode::{EneCode, TopologyGene, NeuronalPropertiesGene, MetaLearningGene, NeuronType};
use evo_rl::graph::FeedForwardNeuralNetwork;

extern crate rand;
use rand::distributions::{Distribution, Bernoulli};

fn main() {
    //Immediate Goal: Evolve a neural network to solve XOR? 

    //1. Define genetic structur for XOR minimal network, including inputs and outputs
    //2. Test random and init over 1000 turns using backprop
    //3. Move this to a unit Test


    let xor_network_genome = EneCode {
        neuron_id: vec!["i01".to_string(), "i02".to_string(), "D".to_string(), "E".to_string()],
        topology: vec![
            TopologyGene {
                innovation_number: "i01".to_string(),
                pin: NeuronType::In,
                inputs: vec![],
                outputs: vec!["D".to_string(), "E".to_string()],
                genetic_weights: vec![],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "i02".to_string(),
                pin: NeuronType::In,
                inputs: vec![],
                outputs: vec!["D".to_string(), "E".to_string()],
                genetic_weights: vec![],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "D".to_string(),
                pin: NeuronType::Hidden,
                inputs: vec!["b".to_string(), "i01".to_string(), "i02".to_string()],
                outputs: vec!["E".to_string()],
                genetic_weights: vec![1., 0.2, 0.3],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "E".to_string(),
                pin: NeuronType::Out,
                inputs: vec!["b".to_string(), "i01".to_string(), "i02".to_string(), "D".to_string()],
                outputs: vec![],
                genetic_weights: vec![1., -0.1, 0.3],
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


    let bernoulli = Bernoulli::new(0.5).unwrap(); // Create a Bernoulli distribution with p=0.5
    let sample: Vec<bool> = (0..2).into_iter().map(|_| bernoulli.sample(&mut rand::thread_rng())).collect();

    let mut fnn = FeedForwardNeuralNetwork::new(xor_network_genome);

    fnn.initialize(); 

    fnn.fwd(vec![0., 1.]);

    let xor_actual = false ^ true;

    let network_evaluation = fnn.fetch_network_output();

    println!("Network evaluation was {} and XOR result was {}!", network_evaluation[0], xor_actual);


}
