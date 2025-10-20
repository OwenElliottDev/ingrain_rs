use reqwest::Client;
use std::collections::HashMap;
use std::error::Error;

pub mod models;
use crate::models::{
    EmbeddingRequest, EmbeddingResponse, GenericMessageResponse, ImageClassificationRequest,
    ImageClassificationResponse, ImageEmbeddingRequest, ImageEmbeddingResponse, LoadModelRequest,
    LoadedModelResponse, MetricsResponse, ModelClassificationLabelsResponse,
    ModelEmbeddingDimsResponse, ModelLibrary, ModelMetadataRequest, RepositoryModelResponse,
    TextEmbeddingRequest, TextEmbeddingResponse, UnloadModelRequest,
};

mod retry;
use crate::retry::retry;

pub struct IngrainClient {
    model_server_url: String,
    inference_server_url: String,
    client: Client,
    retries: u16,
    retry_delay_ms: u64,
}

impl IngrainClient {
    pub fn new(model_server_url: &str, inference_server_url: &str) -> Self {
        IngrainClient {
            model_server_url: model_server_url.to_string(),
            inference_server_url: inference_server_url.to_string(),
            client: Client::new(),
            retries: 0,
            retry_delay_ms: 0,
        }
    }

    pub fn new_with_retries(
        model_server_url: &str,
        inference_server_url: &str,
        retries: u16,
        retry_delay_ms: u64,
    ) -> Self {
        IngrainClient {
            model_server_url: model_server_url.to_string(),
            inference_server_url: inference_server_url.to_string(),
            client: Client::new(),
            retries,
            retry_delay_ms,
        }
    }

    async fn server_health(
        &self,
        api_url: String,
    ) -> Result<GenericMessageResponse, Box<dyn Error>> {
        let response = self.client.get(&api_url).send().await?;

        let status = response.status();

        let parsed_body: GenericMessageResponse = serde_json::from_str(&response.text().await?)?;

        if status.is_success() {
            Ok(parsed_body)
        } else {
            Err(format!("Request failed with status: {}", status).into())
        }
    }

    pub async fn model_server_health(&self) -> Result<GenericMessageResponse, Box<dyn Error>> {
        let api_url = format!("{}/health", self.model_server_url);
        self.server_health(api_url).await
    }

    pub async fn inference_server_health(&self) -> Result<GenericMessageResponse, Box<dyn Error>> {
        let api_url = format!("{}/health", self.inference_server_url);
        self.server_health(api_url).await
    }

    pub async fn loaded_models(&self) -> Result<LoadedModelResponse, Box<dyn Error>> {
        let api_url = format!("{}/loaded_models", self.model_server_url);
        let response = self.client.get(&api_url).send().await?;
        let status = response.status();

        let parsed_body: LoadedModelResponse = serde_json::from_str(&response.text().await?)?;

        if status.is_success() {
            Ok(parsed_body)
        } else {
            Err(format!("Request failed with status: {}", status).into())
        }
    }

    pub async fn repository_models(&self) -> Result<RepositoryModelResponse, Box<dyn Error>> {
        let api_url = format!("{}/repository_models", self.model_server_url);
        let response = self.client.get(&api_url).send().await?;
        let status = response.status();

        let parsed_body: RepositoryModelResponse = serde_json::from_str(&response.text().await?)?;

        if status.is_success() {
            Ok(parsed_body)
        } else {
            Err(format!("Request failed with status: {}", status).into())
        }
    }

    pub async fn metrics(&self) -> Result<MetricsResponse, Box<dyn Error>> {
        let api_url = format!("{}/metrics", self.inference_server_url);
        let response = self.client.get(&api_url).send().await?;
        let status = response.status();

        let parsed_body: MetricsResponse = serde_json::from_str(&response.text().await?)?;

        if status.is_success() {
            Ok(parsed_body)
        } else {
            Err(format!("Request failed with status: {}", status).into())
        }
    }

    pub async fn load_model(
        &self,
        name: String,
        library: ModelLibrary,
    ) -> Result<GenericMessageResponse, Box<dyn Error>> {
        let api_url = format!("{}/load_model", self.model_server_url);

        let payload = LoadModelRequest { name, library };

        let response = self.client.post(api_url).json(&payload).send().await?;

        let status = response.status();
        let body_text = response.text().await?;

        let parsed_body: GenericMessageResponse = serde_json::from_str(&body_text)?;

        if status.is_success() {
            Ok(parsed_body)
        } else {
            Err(format!(
                "Request failed with status: {} and body: {}",
                status, body_text
            )
            .into())
        }
    }

    pub async fn unload_model(
        &self,
        name: String,
    ) -> Result<GenericMessageResponse, Box<dyn Error>> {
        let api_url = format!("{}/unload_model", self.model_server_url);

        let payload = UnloadModelRequest { name };

        let response = self.client.post(api_url).json(&payload).send().await?;

        let status = response.status();
        let body_text = response.text().await?;

        let parsed_body: GenericMessageResponse = serde_json::from_str(&body_text)?;

        if status.is_success() {
            Ok(parsed_body)
        } else {
            Err(format!(
                "Request failed with status: {} and body: {}",
                status, body_text
            )
            .into())
        }
    }

    pub async fn delete_model(
        &self,
        name: String,
    ) -> Result<GenericMessageResponse, Box<dyn Error>> {
        let api_url = format!("{}/delete_model", self.model_server_url);

        let payload = UnloadModelRequest { name };

        let response = self.client.post(api_url).json(&payload).send().await?;

        let status = response.status();
        let body_text = response.text().await?;

        let parsed_body: GenericMessageResponse = serde_json::from_str(&body_text)?;

        if status.is_success() {
            Ok(parsed_body)
        } else {
            Err(format!(
                "Request failed with status: {} and body: {}",
                status, body_text
            )
            .into())
        }
    }

    pub async fn embed_text(
        &self,
        name: String,
        text: Vec<String>,
        normalize: Option<bool>,
        n_dims: Option<u16>,
    ) -> Result<TextEmbeddingResponse, Box<dyn Error>> {
        let api_url = format!("{}/embed_text", self.inference_server_url);

        let payload = TextEmbeddingRequest {
            text,
            normalize,
            n_dims,
            name,
        };

        let request = self.client.post(api_url).json(&payload);

        let response: TextEmbeddingResponse =
            retry(request, self.retries, self.retry_delay_ms).await?;
        Ok(response)
    }

    pub async fn embed_image(
        &self,
        name: String,
        image: Vec<String>,
        normalize: Option<bool>,
        n_dims: Option<u16>,
        image_download_headers: Option<HashMap<String, String>>,
    ) -> Result<ImageEmbeddingResponse, Box<dyn Error>> {
        let api_url = format!("{}/embed_image", self.inference_server_url);

        let payload = ImageEmbeddingRequest {
            image,
            normalize,
            n_dims,
            name,
            image_download_headers,
        };

        let request = self.client.post(api_url).json(&payload);

        let response: ImageEmbeddingResponse =
            retry(request, self.retries, self.retry_delay_ms).await?;
        Ok(response)
    }

    pub async fn embed(
        &self,
        name: String,
        text: Option<Vec<String>>,
        image: Option<Vec<String>>,
        normalize: Option<bool>,
        n_dims: Option<u16>,
        image_download_headers: Option<HashMap<String, String>>,
    ) -> Result<EmbeddingResponse, Box<dyn Error>> {
        if text.is_none() && image.is_none() {
            return Ok(EmbeddingResponse {
                text_embeddings: None,
                image_embeddings: None,
                processing_time_ms: 0.0f32,
            });
        }

        let api_url = format!("{}/embed", self.inference_server_url);

        let payload = EmbeddingRequest {
            image,
            text,
            normalize,
            n_dims,
            name,
            image_download_headers,
        };

        let request = self.client.post(api_url).json(&payload);

        let response: EmbeddingResponse = retry(request, self.retries, self.retry_delay_ms).await?;
        Ok(response)
    }

    pub async fn classify_image(
        &self,
        name: String,
        image: Vec<String>,
        image_download_headers: Option<HashMap<String, String>>,
    ) -> Result<ImageClassificationResponse, Box<dyn Error>> {
        let api_url = format!("{}/classify_image", self.inference_server_url);

        let payload = ImageClassificationRequest {
            image,
            name,
            image_download_headers,
        };

        let request = self.client.post(api_url).json(&payload);

        let response: ImageClassificationResponse =
            retry(request, self.retries, self.retry_delay_ms).await?;
        Ok(response)
    }

    pub async fn model_classification_labels(
        &self,
        name: String,
    ) -> Result<ModelClassificationLabelsResponse, Box<dyn Error>> {
        let api_url = format!("{}/model_classification_labels", self.model_server_url);

        let payload = ModelMetadataRequest { name };

        let request = self.client.get(api_url).query(&payload);

        let response: ModelClassificationLabelsResponse =
            retry(request, self.retries, self.retry_delay_ms).await?;
        Ok(response)
    }

    pub async fn model_embedding_size(
        &self,
        name: String,
    ) -> Result<ModelEmbeddingDimsResponse, Box<dyn Error>> {
        let api_url = format!("{}/model_embedding_size", self.model_server_url);

        let payload = ModelMetadataRequest { name };

        let request = self.client.get(api_url).query(&payload);

        let response: ModelEmbeddingDimsResponse =
            retry(request, self.retries, self.retry_delay_ms).await?;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::{GET, POST};
    use httpmock::MockServer;
    use tokio;

    #[tokio::test]
    async fn test_model_server_health_success() {
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(GET).path("/health");
            then.status(200)
                .header("Content-Type", "application/json")
                .body(r#"{"message": "Model server healthy"}"#);
        });

        let client = IngrainClient::new(&server.url(""), "http://localhost:8686");

        let response = client.model_server_health().await.unwrap();

        assert_eq!(response.message, "Model server healthy");

        mock.assert();
    }

    #[tokio::test]
    async fn test_model_server_health_failure() {
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(GET).path("/health");
            then.status(500)
                .header("Content-Type", "application/json")
                .body(r#"{"message": "Internal Server Error"}"#);
        });

        let client = IngrainClient::new(&server.url(""), "http://localhost:8686");

        let result = client.model_server_health().await;

        assert!(result.is_err());

        mock.assert();
    }

    #[tokio::test]
    async fn test_embed_image_fails_after_retries() {
        let server = MockServer::start();

        // Always fail
        let _fail_mock = server.mock(|when, then| {
            when.method(POST).path("/embed_image");
            then.status(500).body("Internal Error");
        });

        let client =
            IngrainClient::new_with_retries("http://localhost:8687", &server.url(""), 2, 10);

        let result = client
            .embed_image(
                "test-model".to_string(),
                vec!["image_url".to_string()],
                None,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embed_success_no_retry() {
        let server = MockServer::start();

        let success_body = r#"{
            "textEmbeddings": [[0.1, 0.2]],
            "imageEmbeddings": null,
            "processingTimeMs": 7.2
        }"#;

        let _mock = server.mock(|when, then| {
            when.method(POST).path("/embed");
            then.status(200)
                .header("Content-Type", "application/json")
                .body(success_body);
        });

        let client =
            IngrainClient::new_with_retries("http://localhost:8687", &server.url(""), 2, 10);

        let result = client
            .embed(
                "test-model".to_string(),
                Some(vec!["hi".to_string()]),
                None,
                None,
                None,
                None,
            )
            .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.text_embeddings.is_some());
    }
}
