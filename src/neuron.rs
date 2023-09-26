// basic neuron in evorl
// learning based on combination of backprop and hebbian principles
// TODO:
// 1. smooth parameters for non-linearities
// 2. evolvable synaptic learning
extern crate nalgebra as na;

use crate::enecode::NeuronalEneCode;
use na::DVector;


pub struct Nn<'a> {
    pub synaptic_weights: DVector<f32>,
    pub inputs: Vec<&'a str>,
    pub ax: f32,
    pub tau: f32,
    pub learning_threshold: f32,
    pub learning_rate: f32,
    pub alpha: f32,
}

impl<'a> From<NeuronalEneCode<'a>> for Nn<'a> {
    fn from(ene: NeuronalEneCode) -> Self {
        Nn {
            inputs: ene.topology.inputs,
            synaptic_weights: DVector::from_vec(ene.topology.genetic_weights), 
            ax: 0., 
            tau: ene.properties.tau,
            learning_rate: ene.meta.learning_rate,
            learning_threshold: ene.meta.learning_threshold,
            alpha: ene.properties.alpha,
        }
    }
}

impl<'a> Nn<'a> {

    pub fn nonlinearity(&self, z: &f32) -> f32 {
        //using hyperboilc tangent function with parameter alpha
        (z * self.alpha).tanh()
    }


    pub fn fwd(&mut self, input_values: DVector<f32>) {
        let impulse: f32 = input_values.dot(&self.synaptic_weights);
        self.ax = self.ax * (-self.tau).exp() + impulse;
        self.learn();
        self.nonlinearity(&self.ax);

    }

    fn learn(&mut self) {
        //increase synaptic weights in proportion to lerning rate, with ceil
        if self.ax > self.learning_threshold {
            self.synaptic_weights = self.synaptic_weights.map(|x| x + x * self.learning_rate - self.ax * self.learning_rate)
        }
    }

}

