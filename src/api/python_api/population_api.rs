//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Py;
use std::sync::Arc;
use crate::enecode::EneCode;
use crate::population::{Population, PopulationConfig};
use std::path::PathBuf;
use pyo3::exceptions::PyRuntimeError;

#[pyclass]
/// `PopulationApi` is the main entry point from Python to interact with evo_rl. 
///
/// Initialization can proceed either from a simple feedforward network with manually defined
/// numbers of input, output, and hidden units, or from a .json file which specifies the direct
/// encoding of the network (See `enecode.rs)
///
/// Example Python code for evolving a neural network to compute XOR:
/// ```
///import evo_rl
///import logging
///
///FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
///logging.basicConfig(format=FORMAT)
///logging.getLogger().setLevel(logging.INFO)
///
///population_size = 200
///
///configuration = {
///        "population_size": population_size,
///        "survival_rate": 0.2,
///        "mutation_rate": 0.4, 
///        "topology_mutation_rate": 0.4,
///        "input_size": 2,
///        "output_size": 1,
///        "project_name": "xor",
///        "project_directory": "xor_agents"
///        }
///
///
///def evaluate_xor_fitness(population_api, agent_idx):
///    fitness = 6
///
///    complexity = population_api.agent_complexity(agent_idx)
///    complexity_penalty = 0.1 * complexity
///
///    for bit1 in [0, 1]:
///        for bit2 in [0, 1]:
///            population_api.agent_fwd(agent_idx, [bit1, bit2])
///            network_output = population_api.agent_out(agent_idx)
///            xor_true = (bit1 > 0) ^ (bit2 > 0)
///            fitness -= (xor_true - network_output[0])**2
///
///    if fitness > complexity_penalty:
///        fitness_eval = fitness - complexity_penalty
///    else:
///        fitness_eval = 0
///
///    population_api.set_agent_fitness(agent_idx, fitness_eval)
///    return 
///
///p = evo_rl.PopulationApi(configuration)
///
///while p.generation < 1000:
///
///    if p.fitness >= 5.8:
///        break
///
///    for agent in range(population_size):
///        evaluate_xor_fitness(p, agent)
///
///    p.update_population_fitness()
///    p.report()
///    p.evolve_step()
/// ```
///
/// Networks can be visualized using GraphViz from `.dot` files saved in the project directory. 
///
/// Furthermore the genome for a particular agent in the population can be written using the
/// `agent_checkpt` method of `PopulationApi` and used as the starting point for evolution for a new
/// or a variant of the same task.
pub struct PopulationApi {
    population: Box<Population>,
    evolve_config: Box<PopulationConfig>,
    pop_config: Py<PyDict>
}

#[pymethods]
impl PopulationApi {
    #[new]
    ///Population constructor which can either utilize inputs/outputs in the Configuration
    ///dictionary as a starting point (hidden units are the same as the number of inputs), or can
    ///use a `.json` file that encodes an Enecode struct. 
    pub fn new(pyconfig: Py<PyDict>, checkpoint: Option<PathBuf>) -> PyResult<Self> {


        let population = Python::with_gil(|py| -> PyResult<Population> {

            let config: &PyDict = pyconfig.as_ref(py);

            let input_size: usize = match config.get_item("input_size")? {
                Some(x) => x.extract()?,
                None => panic!("Input size for network is not defined.")
            };

            let hidden_size: usize = match config.get_item("hidden_size")? {
                Some(x) => x.extract()?,
                None => panic!("Hidden unit size for network is not defined.")
            };

            let output_size: usize = match config.get_item("output_size")? {
                Some(x) => x.extract()?,
                None => panic!("Input size for network is not defined.")
            };

            let population_size: usize = match config.get_item("population_size")? {
                Some(x) => x.extract()?,
                None => panic!("missing population size parameter")
            };

            let survival_rate: f32  = match config.get_item("survival_rate")? {
                Some(x) => x.extract()?,
                None => panic!("missing population survival rate parameter")
            };

            let mutation_rate: f32  = match config.get_item("mutation_rate")? {
                Some(x) => x.extract()?,
                None => panic!("missing population mutation rate parameter")
            };

            let topology_mutation_rate: f32  = match config.get_item("topology_mutation_rate")? {
                Some(x) => x.extract()?,
                None => panic!("missing population topology rate parameter")
            };

            let network_module: Option<String> = match config.get_item("network_module")? {
                Some(x) => Some(x.extract()?),
                None => None
            };


            let genome: EneCode = match checkpoint {
                Some(chkpt) => match EneCode::try_from(&chkpt) {
                    Ok(enecode) => enecode,
                    Err(err) => panic!("{}", err)
                },
                None => EneCode::new(input_size, hidden_size, output_size, network_module.as_deref())
            };


            Ok(Population::new(genome, population_size, survival_rate, mutation_rate, topology_mutation_rate))

        })?;

        let evolve_config = Python::with_gil(|py| -> PyResult<PopulationConfig> {

            let config: &PyDict = pyconfig.as_ref(py);

            let project_name: String = match config.get_item("project_name")? {
                Some(x) => x.extract()?,
                None => panic!("missing project name")
            };

            let project_directory: String = match config.get_item("project_directory")? {
                Some(x) => x.extract()?,
                None => panic!("missing project directory")
            };

            let project_path: Arc<PathBuf> = Arc::from(PathBuf::from(project_directory));

            Ok(PopulationConfig::new(Arc::from(project_name), Some(project_path), None, 200, 0.50, 0.50, true, None))

        })?;


        Ok(PopulationApi {
            population: Box::new(population),
            evolve_config: Box::new(evolve_config),
            pop_config: pyconfig
        })
    }

    /// Each round of selection calls this method to evaluate fitness.
    pub fn evolve_step(&mut self) {
        self.population.evolve_step(&self.evolve_config);
    }

    /// This method needs to be called manually to update the vector of fitness scores for each
    /// agent
    pub fn update_population_fitness(&mut self) {
        self.population.update_population_fitness();
    }

    /// Reports the average fitness of the population, best agent fitness, and id of the best
    /// agent. 
    pub fn report(&self) {
        self.population.report(&self.evolve_config);
    }

    /// Evaluates an agent's otuput given an input vector.
    pub fn agent_fwd(&mut self, idx: usize, input: Py<PyList>) {
        let py_vec = Python::with_gil(|py| -> Result<Vec<f32>, PyErr> {
            let input_vec = input.as_ref(py);
            input_vec.iter()
                .map(|p| p.extract::<f32>())
                .collect()
            });

        match py_vec {
            Ok(v) => self.population.agents[idx].fwd(v),
            err => error!("PyError: {:?}", err)
        }

    }

    /// Serializes the agent's genome as a `.json` file and writes it to disk. This can also be
    /// used as the starting point for evolution. 
    pub fn agent_checkpt(&self, idx: usize, file_path: PathBuf) -> PyResult<()> {
        let write_success = self.population.write_agent_genome(idx, file_path);
        let py_result = match write_success {
            Ok(value) => Ok(value),
            Err(err) => Err(PyRuntimeError::new_err(format!("{}", err)))
        };

        py_result
    }

    /// Gets the agent's current output value. 
    pub fn agent_out(&self, idx: usize) -> PyResult<Vec<f32>> {
        Ok(self.population.agents[idx].output())
    }

    /// Returns a metric of agent complexity which currently is a count of how many neurons it has. 
    pub fn agent_complexity(&self, idx: usize) -> PyResult<usize> {
        Ok(self.population.agents[idx].nn.node_identity_map.len())
    }

    /// Sets the agent's fitness. This is necessary due to Rust's restrictions on safety. 
    pub fn set_agent_fitness(&mut self, idx: usize, value: f32) {
        self.population.agents[idx].fitness = value;
    }

    #[getter(generation)]
    fn generation(&self) -> PyResult<usize> {
        Ok(self.population.generation)
    }

    #[getter(fitness)]
    fn fitness(&self) -> PyResult<f32> {
        Ok(self.population.population_fitness)
    }

    fn deserialize_enecode(&self, agent_checkpoint: PathBuf) -> PyResult<EneCode> {
        let genome = EneCode::try_from(&agent_checkpoint);

        let py_result = match genome {
            Ok(value) => Ok(value),
            Err(err) => Err(PyRuntimeError::new_err(format!("{}", err)))
        };

        py_result
    }


}

