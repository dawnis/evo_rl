//!This module implements a struct which manages the quality-diversity database associated with a
//!particular module.

use reqwest::Client;
use std::error::Error;
use std::sync::Arc;
use std::collections::HashMap;

use crate::agent_wrapper::Agent;
use crate::enecode::EneCode;


//TODO: MVP is to keep track of a single genome parameterized as X=0 and y=0
pub struct QDManager {
   module: Arc<str>,
   endpoint: Arc<str>,
   library: HashMap<(i32, i32), EneCode>

}

async fn fetch_genome(full_api_endpoint: &str, module: &str, location: (i32, i32)) -> Result<EneCode, Box<dyn Error>> {

        let r_client = Client::new();
    
        let response = r_client
            .get(full_api_endpoint)
            .query(&[("module", module), ("param_x", &location.0.to_string()), ("param_y", &location.1.to_string())])
            .send()
            .await?;
    
        if response.status().is_success() {
            let genome: EneCode = response.json().await?;
            Ok(genome)
        } else {
            Err(format!("Request failed with status: {}", response.status()).into())
        }
}


impl QDManager {

    pub fn new(module: Arc<str>, endpoint: Arc<str>) -> Self {
        QDManager {
            module,
            endpoint,
            library: HashMap::new(),
        }
    }

    pub fn poplation_vector(population_size: usize) -> Vec<Agent> {
        //TODO: for now assume that the module has entries in the database. If it is new,
        //will deal with this later
    }


    // pub fn post_genome(&self, full_api_endpoint: &str) -> Result<String, Box<dyn Error>> {
    //     let r_client = Client::new();
    //     let serialized_genome = self.nn.serialize_genome();
    //
    //     let response = r_client
    //         .post(full_api_endpoint)
    //         .json(&serialized_genome)
    //         .send()?;
    //
    //     if response.status().is_success() {
    //         let body = response.text()?;
    //         Ok(body)
    //     } else {
    //         Err(format!("Request failed with status: {}", response.status()).into())
    //     }
    // }
}
