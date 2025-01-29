//!This module implements a struct which manages the quality-diversity database associated with a
//!particular module.

use reqwest::{Client, Url};
use std::error::Error;
use std::sync::Arc;
use std::collections::HashMap;

use crate::agent_wrapper::Agent;
use crate::enecode::EneCode;


//TODO: MVP is to keep track of a single genome parameterized as X=0 and y=0
pub struct QDManager {
   module: Arc<str>,
   endpoint: Option<Url>,
   qdlib: HashMap<(i32, i32), EneCode>

}

async fn fetch_genome(full_api_endpoint: Url, module: &str, location: (i32, i32)) -> Result<EneCode, Box<dyn Error>> {

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

    pub fn new_from_genome(module: Arc<str>, endpoint: Option<Url>, genome_base: EneCode) -> Self {
        let mut qdlib: HashMap<(i32, i32), EneCode> = HashMap::new();
        qdlib.insert((0, 0), genome_base);

        QDManager {
            module,
            endpoint,
            qdlib,
        }

    }

    pub async fn new(module: Arc<str>, endpoint: Option<Url>) -> Self {

        let endpt = endpoint.clone();

        let api = match endpt {
            Some(url) => url,
            None => panic!("Expected a valid url for qdm connection!")
        };
        

        let seed = match fetch_genome(api, &module, (0, 0)).await {
            Ok(s) => s,
            Err(e) => panic!("Error fetching genome: {}", e)

        };

        let mut qdlib: HashMap<(i32, i32), EneCode> = HashMap::new();
        qdlib.insert((0, 0), seed);

        Self {
            module,
            endpoint,
            qdlib,
        }
    }

    pub async fn postg(&self) {
        //Posts genome into Postgres db
    }

    pub fn fetchg(&self, location: (i32, i32)) -> &EneCode {
        match self.qdlib.get(&location) {
            Some(g) => g,
            None => panic!("Location not represented in library")
        }
    }

    pub fn gen_agent_vector(&self, vector_size: usize) -> Vec<Agent> {

        let mut agent_vector:Vec<Agent> = Vec::new();
        let genome_base = self.fetchg((0,0));
        
        
        for _idx in 0..vector_size {
            let mut agent = Agent::new(genome_base.clone());
        
            //Random mutation of newly initialized population members
            agent.mutate(1., 10., 0.);
            agent_vector.push(agent);
        }

        agent_vector
    }

    // pub fn poplation_vector(population_size: usize) -> Vec<Agent> {
    //     //TODO: for now assume that the module has entries in the database. If it is new,
    //     //will deal with this later
    // }


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
