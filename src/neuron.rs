// basic neuron in evorl
// learning based on combination of backprop and hebbian principles
// TODO:
// 1. smooth parameters for non-linearities
// 2. evolvable synaptic learning
extern crate nalgebra as na;

use crate::enecode::{NeuronalEneCode, NeuronType};
use na::{DVector, Vector1};


#[derive(Debug, Clone)]
pub struct Nn<'a> {
    pub synaptic_weights: DVector<f32>,
    pub inputs: Vec<&'a str>,
    pub output: f32,
    pub tau: f32,
    pub learning_threshold: f32,
    pub learning_rate: f32,
    pub alpha: f32,
    pub neuron_type: NeuronType,
}

impl<'a> From<NeuronalEneCode<'a>> for Nn<'a> {
    fn from(ene: NeuronalEneCode<'a>) -> Self {
        Nn {
            inputs: ene.topology.inputs.clone(),
            synaptic_weights: DVector::from_vec(ene.topology.genetic_weights.clone()), 
            output: 0., 
            tau: ene.properties.tau,
            learning_rate: ene.meta.learning_rate,
            learning_threshold: ene.meta.learning_threshold,
            alpha: ene.properties.alpha,
            neuron_type: ene.topology.pin,
        }
    }
}

impl<'a> Nn<'a> {

    pub fn propagate(&mut self, input: DVector<f32>) -> f32 {
        match self.neuron_type {
            NeuronType::In => self.fwd_input(input),
            NeuronType::Bias => self.fwd_input(input),
            NeuronType::Hidden => self.fwd(input),
            NeuronType::Out => self.fwd(input),
        }
    }

    fn fwd_input(&mut self, in_value: DVector<f32>) -> f32 {
        in_value[0]
    }

    fn fwd(&mut self, input_values: DVector<f32>) -> f32 {
        let impulse: f32 = input_values.dot(&self.synaptic_weights);
        self.output = self.output * (-self.tau).exp() + impulse;
        self.learn();
        self.nonlinearity(&self.output)

    }

    fn nonlinearity(&self, z: &f32) -> f32 {
        //using hyperboilc tangent function with parameter alpha
        (z * self.alpha).tanh()
    }


    fn learn(&mut self) {
        //increase synaptic weights in proportion to lerning rate, with ceil
        if self.output > self.learning_threshold {
            self.synaptic_weights = self.synaptic_weights.map(|x| x + x * self.learning_rate - self.output * self.learning_rate)
        }
    }

}

