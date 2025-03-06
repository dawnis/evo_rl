//!This module implements a struct which manages the quality-diversity database associated with a
//!particular module.

use log::*;

use reqwest::blocking::{Client, Response};
use reqwest::header::CONTENT_TYPE;
use reqwest::{Error, Url};

use serde_json::to_string;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::agent_wrapper::Agent;
use crate::enecode::EneCode;

pub struct QDManager {
    module: Arc<str>,
    endpoint: Option<Url>,
    qdlib: HashMap<(i32, i32), EneCode>,
}

fn fetch_genome(
    full_api_endpoint: Url,
    module: &str,
    location: (i32, i32),
) -> Result<EneCode, Error> {
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;

    let genome: EneCode = client
        .get(full_api_endpoint)
        .query(&[
            ("module", module),
            ("param_x", &location.0.to_string()),
            ("param_y", &location.1.to_string()),
        ])
        .send()?
        .json()?;

    Ok(genome)
}

fn post_genome(
    full_api_endpoint: Url,
    module: &Arc<str>,
    location: (i32, i32),
    genome: &EneCode,
) -> Result<Response, Error> {
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;


    let modulen: &str = &module;
    let param_x = location.0.to_string();
    let param_y = location.1.to_string();

    let mut params = HashMap::new();

    params.insert("module", modulen);
    params.insert("x", &param_x);
    params.insert("y", &param_y);

    // Serialize the genome into a JSON string for logging purposes
    match to_string(&genome) {
        Ok(genome_json) => {
            // Log the serialized genome as debug information
            info!("Request Body: {:#?}", genome_json);
        }
        Err(e) => {
            error!("Failed to serialize genome: {}", e);
        }
    }


    let result = client
        .post(full_api_endpoint)
        .query(&params)
        .header(CONTENT_TYPE, "application/json")
        .json(&genome)
        .send();

    

    //TODO: Log or print the request body on both the Rust and Scala sides to verify that the serialized JSON is correctly formatted and sent.

    match result {
        Ok(response) => {
            info!("Ok response {}", response.status());
            Ok(response)
            }
        Err(e) => {
            if e.is_timeout() {
                error!("Timeout error!")
            } else {
                error!("Response error {}", e);
            }
            Err(e)
        }
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

    pub fn new(module: Arc<str>, endpoint: Option<Url>) -> Self {
        Self {
            module,
            endpoint,
            qdlib: HashMap::new(),
        }
    }

    pub fn init_library(&self) {
        if self.qdlib.is_empty() {
            self.api_fetch_library();
        }
    }

    //Fetches genomes associated with module from parameter (0, 0) in the postgres database
    pub fn api_fetch_library(&self) {
        let apiurl = match &self.endpoint {
            Some(url) => url,
            None => panic!("No api endpoint set but asked to fetch library"),
        };

        let seed = match fetch_genome(apiurl.clone(), &self.module, (0, 0)) {
            Ok(s) => s,
            Err(e) => panic!("Error fetching genome: {}", e),
        };

        let mut qdlib: HashMap<(i32, i32), EneCode> = HashMap::new();
        qdlib.insert((0, 0), seed);
    }

    pub fn postg(&self, genome: &EneCode) -> Result<Response, Error> {

        let apiurl = match &self.endpoint {
            Some(url) => url,
            None => panic!("No api endpoint set but asked to post genome"),
        };

        info!("attempting to post genome!!");

        post_genome(apiurl.clone(), &self.module, (0, 0), genome)
    }

    pub fn fetchg(&self, location: (i32, i32)) -> &EneCode {
        match self.qdlib.get(&location) {
            Some(g) => g,
            None => panic!("Location not represented in library"),
        }
    }

    pub fn gen_agent_vector(&self, vector_size: usize) -> Vec<Agent> {
        let mut agent_vector: Vec<Agent> = Vec::new();
        let genome_base = self.fetchg((0, 0));

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


#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Error;
    use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};
    use httpmock::prelude::*;
    use crate::setup_logger;

    #[test]
    fn test_fetch_genome_error() {

        let server = MockServer::start();

        let genome = GENOME_EXAMPLE.clone();

        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/genome")
                .query_param("module", "test_module")
                .query_param("param_x", "0")
                .query_param("param_y", "0");

            then.status(200)
                .header("content-type", "application/json")
                .json_body_obj(&genome);
        });

        let uri = format!("{}/genome", server.base_url());
        let api: Url = Url::parse(&uri).expect("Url Parse Error");

        // let qdm: QDManager = QDManager::new(Arc::from("testmodule"), Some(api));

        let result = fetch_genome(api, "nonexistent_test_module", (0,0));

        assert!(result.is_err());
    }

    #[test]
    fn test_fetch_genome_success() {
        setup_logger();

        let server = MockServer::start();

        let genome = GENOME_EXAMPLE.clone();

        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/genome")
                .query_param("module", "test_module")
                .query_param("param_x", "0")
                .query_param("param_y", "0");

            then.status(200)
                .header("content-type", "application/json")
                .json_body_obj(&genome);
        });

        let uri = format!("{}/genome", server.base_url());
        let api: Url = Url::parse(&uri).expect("Url Parse Error");

        // let qdm: QDManager = QDManager::new(Arc::from("testmodule"), Some(api));
        

        let result = fetch_genome(api, "test_module", (0,0)).expect("httpmock failure");

        debug!("{:?}", result);

        assert_eq!(result, genome);
    }

    #[test]
    fn test_post_genome_success() {
        setup_logger();

        let server = MockServer::start();

        let genome = GENOME_EXAMPLE.clone();

        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/genome")
                .query_param("module", "test_module")
                .query_param("x", "0")
                .query_param("y", "0")
                .json_body_obj(&genome);
                

            then.status(200);

        });

        let uri = format!("{}/genome", server.base_url());
        let api: Url = Url::parse(&uri).expect("Url Parse Error");

        // let qdm: QDManager = QDManager::new(Arc::from("testmodule"), Some(api));
        

        let response = post_genome(api,
            &Arc::from("test_module"),
            (0,0),
            &genome).unwrap();

        debug!("{:?}", response);
        
        mock.assert();

        assert_eq!(response.status(), 200);
    }
}
