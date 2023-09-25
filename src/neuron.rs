// basic neuron in evorl
// learning based on combination of backprop and hebbian principles
// TODO:
// 1. smooth parameters for non-linearities
// 2. evolvable synaptic learning
use nalgebra::Vector3;

pub struct Nn {
    pub syn: Vector3<f32>,
    pub ax: f32,
    pub tau: f32,
    pub learning_threshold: f32,
    pub learning_rate: f32,
    pub alpha: f32,
}

impl Nn {

    pub fn nonlinearity(&self, z: &f32) -> f32 {
        //using hyperboilc tangent function with parameter alpha
        (z * self.alpha).tanh()
    }


    pub fn fwd(&mut self, input: Vector3<f32>) {
        let impulse: f32 = input.dot(&self.syn);
        self.ax = self.ax * (-self.tau).exp() + impulse;
        self.learn();
        self.nonlinearity(&self.ax);

    }

    fn learn(&mut self) {
        //increase synaptic weights in proportion to lerning rate, with ceil
        if self.ax > self.learning_threshold {
            self.syn = self.syn.map(|x| x + x * self.learning_rate - self.ax * self.learning_rate)
        }
    }



}
