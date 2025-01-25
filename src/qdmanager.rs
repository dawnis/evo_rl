//!This module implements a struct which manages the quality-diversity database associated with a
//!particular module.

use reqwest::Client;
use std::error::Error;

pub struct QDManager {}

impl QDManager {
    pub fn post_genome(&self, full_api_endpoint: &str) -> Result<String, Box<dyn Error>> {
        let r_client = Client::new();
        let serialized_genome = self.nn.serialize_genome();

        let response = r_client
            .post(full_api_endpoint)
            .json(&serialized_genome)
            .send()?;

        if response.status().is_success() {
            let body = response.text()?;
            Ok(body)
        } else {
            Err(format!("Request failed with status: {}", response.status()).into())
        }
    }
}
