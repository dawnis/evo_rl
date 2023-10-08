use evo_rl::{graph::NeuralNetwork, enecode::EneCode};
use evo_rl::doctest::XOR_GENOME;

extern crate rand;
use rand::distributions::{Distribution, Bernoulli};

fn main() {
    //Immediate Goal: Evolve a neural network to solve XOR? 

    //1. Define genetic structur for XOR minimal network, including inputs and outputs
    //2. Test random and init over 1000 turns using backprop
    //3. Move this to a unit Test


    let bernoulli = Bernoulli::new(0.5).unwrap(); // Create a Bernoulli distribution with p=0.5
    let sample: Vec<bool> = (0..2).into_iter().map(|_| bernoulli.sample(&mut rand::thread_rng())).collect();

    let xor_genome: EneCode = XOR_GENOME.clone();
    let mut fnn = NeuralNetwork::new(xor_genome);

    fnn.initialize(); 

    fnn.fwd(vec![0., 1.]);

    let xor_actual = false ^ true;

    let network_evaluation = fnn.fetch_network_output();

    println!("Network evaluation was {} and XOR result was {}!", network_evaluation[0], xor_actual);


}
