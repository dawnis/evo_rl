// basic neuron in evorl
// learning based on combination of backprop and hebbian principles
// TODO:
// 1. smooth parameters for non-linearities
// 2. evolvable synaptic learning
extern crate nalgebra as na;

use crate::enecode::{NeuronalEneCode, NeuronType};
use na::DVector;


#[derive(Debug, Clone)]
pub struct Nn {
    pub synaptic_weights: DVector<f32>,
    pub synaptic_bias: f32,
    pub inputs: Vec<String>,
    pub activation_level: f32,
    pub tau: f32,
    pub learning_threshold: f32,
    pub learning_rate: f32,
    pub tanh_alpha: f32,
    pub neuron_type: NeuronType,
}

impl From<NeuronalEneCode> for Nn {
    fn from(ene: NeuronalEneCode) -> Self {
        Nn {
            inputs: ene.topology.inputs.clone(),
            synaptic_weights: DVector::from_vec(ene.topology.genetic_weights.clone()), 
            synaptic_bias: ene.topology.genetic_bias,
            activation_level: 0., 
            tau: ene.properties.tau,
            learning_rate: ene.meta.learning_rate,
            learning_threshold: ene.meta.learning_threshold,
            tanh_alpha: ene.properties.tanh_alpha,
            neuron_type: ene.topology.pin,
        }
    }
}

impl Nn{

    //computes next state of neuron 
    pub fn propagate(&mut self, input: DVector<f32>) {
        match self.neuron_type {
            NeuronType::In => {
                assert_eq!(input.len(), 1);
                self.set_value(input[0]);
            },
            _ => self.fwd(input),
        }
    }

    //returns the output value of the neuron
    pub fn output_value(&self) -> f32 {
        match self.neuron_type {
            NeuronType::In => self.activation_level,
            _ => self.nonlinearity(&self.activation_level)

        }
    }

    fn set_value(&mut self, in_value: f32) {
        self.activation_level = in_value;
    }

    fn fwd(&mut self, input_values: DVector<f32>) {
        let impulse: f32 = input_values.dot(&self.synaptic_weights);
        self.activation_level = self.activation_level * (-self.tau).exp() + impulse + self.synaptic_bias;
        self.learn();
    }

    fn nonlinearity(&self, z: &f32) -> f32 {
        //using hyperboilc tangent function with parameter alpha
        (z * self.tanh_alpha).tanh()
    }


    fn learn(&mut self) {
        //increase synaptic weights in proportion to lerning rate, with ceil
        if self.activation_level > self.learning_threshold {
            self.synaptic_weights = self.synaptic_weights.map(|x| x + x * self.learning_rate - self.activation_level * self.learning_rate)
        }
    }

}

