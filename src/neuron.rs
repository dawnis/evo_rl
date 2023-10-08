// TODO:
// 1. backpropagation 
// 2. evolvable synaptic learning

use nalgebra as na;
use std::sync::Arc;
use crate::enecode::{NeuronalEneCode, NeuronType};
use na::DVector;


//// `Nn` is a struct that defines an Artificial Neuron.
/// 
/// # Fields
/// * `synaptic_weights` - A vector of synaptic weights.
/// * `neuronal_bias` - A synaptic bias for the neuron.
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
    pub id: String,
    pub synaptic_weights: DVector<f32>,
    pub neuronal_bias: f32,
    pub inputs: Vec<String>,
    pub activation_level: f32,
    pub tau: f32,
    pub learning_threshold: f32,
    pub learning_rate: f32,
    pub tanh_alpha: f32,
    pub neuron_type: NeuronType,
}

impl From<Arc<NeuronalEneCode<'_>>> for Nn {
    fn from(ene: Arc<NeuronalEneCode>) -> Self {
        let mut inputs_as_list: Vec<String> = Vec::new();
        let mut weights_as_list: Vec<f32> = Vec::new();

        for input_id in ene.topology.inputs.keys() {
            inputs_as_list.push(input_id.clone());
            weights_as_list.push(ene.topology.inputs[input_id]);
        }

        Nn {
            id: ene.neuron_id.clone(),
            inputs: inputs_as_list,
            synaptic_weights: DVector::from_vec(weights_as_list), 
            neuronal_bias: ene.topology.genetic_bias,
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
    /// * `input` -f32
    ///
    /// # Example
    /// ```rust
    /// // code example here
    /// ```
    pub fn propagate(&mut self, input: f32) {
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
    ///
    /// # Example
    /// ```rust
    /// // code example here
    /// ```
    pub fn output_value(&self) -> f32 {
        match self.neuron_type {
            NeuronType::In => self.activation_level,
            _ => self.nonlinearity(&self.activation_level)

        }
    }

    fn set_value(&mut self, in_value: f32) {
        self.activation_level = in_value;
        println!("Setting neuron {} to activation level of {}", self.id, self.activation_level);
    }

    fn fwd(&mut self, impulse: f32) {
        self.activation_level = self.activation_level * (-self.tau).exp() + impulse + self.neuronal_bias;
        //self.learn?
        println!("Activation level for neuron {} set at {} after impulse {}", self.id, self.activation_level, impulse);
    }

    fn nonlinearity(&self, z: &f32) -> f32 {
        //using hyperboilc tangent function with parameter alpha
        (z * self.tanh_alpha).tanh()
    }


    fn learn(&self, syn_weight_current: f32) -> f32 {
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
            neuron_id: "h01".to_string(),
            topology: &TOPOLOGY_GENE_EXAMPLE,
            properties: &NEURONAL_PROPERTIES_GENE_EXAMPLE,
            meta: &META_GENE_EXAMPLE,
        };

        let mut neuron = Nn::from(Arc::new(nec));
        neuron.propagate(12_f32);

        assert_eq!(neuron.activation_level, 17.);

    }

    #[test]
    fn test_output_value() {
        let nec = NeuronalEneCode {
            neuron_id: "h01".to_string(),
            topology: &TOPOLOGY_GENE_EXAMPLE,
            properties: &NEURONAL_PROPERTIES_GENE_EXAMPLE,
            meta: &META_GENE_EXAMPLE,
        };

        let mut neuron = Nn::from(Arc::new(nec));
        neuron.propagate(12_f32);

        assert_eq!(neuron.activation_level, 17.);
        assert_eq!(neuron.output_value(), (17_f32).tanh());
    }
}

