//! Module for the `neuron` struct which defines the individual unit of computation for the neural
//! network. 

use log::*;
use nalgebra as na;
use std::sync::Arc;
use crate::enecode::{NeuronalEneCode, NeuronType};
use na::DVector;
use rand::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::{relu, sigmoid};

//// `Nn` is a struct that defines an Artificial Neuron.
/// 
/// # Fields
/// * `synaptic_weights` - A vector of synaptic weights.
/// * `bias` - A synaptic bias for the neuron.
/// * `inputs` - A list of input neurons.
/// * `activation_level` - The neuron's activation level.
/// * `tau` - The neuron's time constant.
/// * `learning_threshold` - The neuron's learning threshold.
/// * `learning_rate` - The neuron's learning rate.
/// * `tanh_alpha` - The neuron's hyperbolic tangent alpha parameter.
/// * `neuron_type` - The type of the neuron: Input, Output, or Hidden.
///
/// # Example
/// ```rust
/// // code example here
#[derive(Debug, Clone)]
pub struct Nn {
    pub id: Arc<str>,
    pub synaptic_weights: DVector<f64>,
    pub bias: f64,
    pub inputs: Vec<String>,
    pub activation_level: f64,
    pub tau: f64,
    pub learning_threshold: f64,
    pub learning_rate: f64,
    pub tanh_alpha: f64,
    pub neuron_type: NeuronType,
}

impl From<Arc<NeuronalEneCode<'_>>> for Nn {
    fn from(ene: Arc<NeuronalEneCode>) -> Self {
        let mut inputs_as_list: Vec<String> = Vec::new();
        let mut weights_as_list: Vec<f64> = Vec::new();

        for input_id in ene.topology.inputs.keys() {
            inputs_as_list.push(input_id.clone());
            weights_as_list.push(ene.topology.inputs[input_id]);
        }

        Nn {
            id: ene.neuron_id.clone(),
            inputs: inputs_as_list,
            synaptic_weights: DVector::from_vec(weights_as_list), 
            bias: ene.topology.genetic_bias,
            activation_level: 0., 
            tau: ene.properties.tau,
            learning_rate: ene.meta.learning_rate,
            learning_threshold: ene.meta.learning_threshold,
            tanh_alpha: ene.properties.tanh_alpha,
            neuron_type: ene.topology.pin.clone(),
        }
    }
}

impl Nn{

    /// Propagates the input through the neuron to compute the next state.
    ///
    /// # Arguments
    /// * `input` -f64
    pub fn propagate(&mut self, input: f64) {
        match self.neuron_type {
            NeuronType::In => {
                self.set_value(input);
            },
            _ => self.fwd(input),
        }
    }

    /// Returns the output value of the neuron.
    ///
    /// # Returns
    /// The output value as a floating-point number.
    pub fn output_value(&self) -> f64 {
        match self.neuron_type {
            NeuronType::In => self.activation_level,
            _ => self.nonlinearity(&self.activation_level)

        }
    }

    /// Performs mutation on the neuron
    ///
    /// # Arguments
    /// * `rng` - thread_rng
    /// * `epsilon` - mutation rate
    /// * `sd` - the standard deviation of a normal distribution used to sample changes
    pub fn mutate<R: Rng>(&mut self, rng: &mut R, epsilon: f64, sd: f64) {
        //bias mutation
        let normal = Normal::new(0., sd).unwrap();
        if rng.gen::<f64>() < epsilon {
            let updated_bias = self.bias + normal.sample(rng);
            self.bias = updated_bias;
        }
    }

    fn set_value(&mut self, in_value: f64) {
        self.activation_level = in_value;
        debug!("Setting neuron {} to activation level of {}", self.id, self.activation_level);
    }

    fn fwd(&mut self, impulse: f64) {
        self.activation_level = self.activation_level - self.activation_level*(-self.tau).exp();
        self.activation_level += impulse + self.bias;
        //self.learn?
        debug!("Activation level for neuron {} set at {} after impulse {}", self.id, self.activation_level, impulse);
    }

    fn nonlinearity(&self, z: &f64) -> f64 {
        // Use relu on hidden layers, tanh on output
        match self.neuron_type {
         NeuronType::Hidden => relu(z),
         _ => (z * self.tanh_alpha).tanh()
        }
    }


    fn learn(&self, syn_weight_current: f64) -> f64 {
        //Calculates a delta to change the current synapse
        if self.activation_level > self.learning_threshold {
            syn_weight_current * self.learning_rate // - self.activation_level * self.homeostatic_force
        } else { 0.} 
    }

}

#[cfg(test)]
mod tests {
    use crate::enecode::*;
    use crate::doctest::{TOPOLOGY_GENE_EXAMPLE, META_GENE_EXAMPLE, NEURONAL_PROPERTIES_GENE_EXAMPLE};
    use super::*;

    #[test]
    fn test_propagate_euron() {
        // Create a NeuronalEneCode and use it to initialize an Nn with NeuronType::Hidden
        //
        let nec = NeuronalEneCode {
            neuron_id: "h01".into(),
            topology: &TOPOLOGY_GENE_EXAMPLE,
            properties: &NEURONAL_PROPERTIES_GENE_EXAMPLE,
            meta: &META_GENE_EXAMPLE,
        };

        let mut neuron = Nn::from(Arc::new(nec));
        neuron.propagate(12_f64);

        assert_eq!(neuron.activation_level, 17.);

    }

    #[test]
    fn test_output_value() {
        let nec = NeuronalEneCode {
            neuron_id: "h01".into(),
            topology: &TOPOLOGY_GENE_EXAMPLE,
            properties: &NEURONAL_PROPERTIES_GENE_EXAMPLE,
            meta: &META_GENE_EXAMPLE,
        };

        let mut neuron = Nn::from(Arc::new(nec));
        neuron.propagate(12_f64);

        assert_eq!(neuron.activation_level, 17.);
        assert_eq!(neuron.output_value(), relu(&17.));

        //multiple runs of the same neuron with 0 tau should produce the same value
        //in the absence of synaptic learning

        neuron.propagate(12_f64);
        assert_eq!(neuron.activation_level, 17.);
        assert_eq!(neuron.output_value(), relu(&17.));
    }

    #[test]
    fn test_mutate() {
        let nec = NeuronalEneCode {
            neuron_id: "h01".into(),
            topology: &TOPOLOGY_GENE_EXAMPLE,
            properties: &NEURONAL_PROPERTIES_GENE_EXAMPLE,
            meta: &META_GENE_EXAMPLE,
        };

        let mut neuron = Nn::from(Arc::new(nec));

        let seed = [17; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);
        neuron.mutate(&mut rng, 1., 0.1);

        assert_ne!(neuron.bias, 5.);
    }
}

