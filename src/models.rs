use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingRequest {
    pub name: String,
    pub text: Option<Vec<String>>,
    pub image: Option<Vec<String>>,
    pub normalize: Option<bool>,
    pub n_dims: Option<u16>,
    pub image_download_headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextEmbeddingRequest {
    pub name: String,
    pub text: Vec<String>,
    pub normalize: Option<bool>,
    pub n_dims: Option<u16>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageEmbeddingRequest {
    pub name: String,
    pub image: Vec<String>,
    pub normalize: Option<bool>,
    pub n_dims: Option<u16>,
    pub image_download_headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageClassificationRequest {
    pub name: String,
    pub image: Vec<String>,
    pub image_download_headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLibrary {
    OpenClip,
    SentenceTransformers,
    Timm,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub name: String,
    pub library: ModelLibrary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnloadModelRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadataRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenericMessageResponse {
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadedModel {
    pub name: String,
    pub library: ModelLibrary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadedModelResponse {
    pub models: Vec<LoadedModel>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RepositoryModel {
    pub name: String,
    pub state: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RepositoryModelResponse {
    pub models: Vec<RepositoryModel>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceStats {
    pub count: Option<String>,
    pub ns: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchStats {
    pub batch_size: String,
    pub compute_input: InferenceStats,
    pub compute_infer: InferenceStats,
    pub compute_output: InferenceStats,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStats {
    pub name: String,
    pub version: String,
    pub inference_stats: HashMap<String, InferenceStats>,
    pub last_inference: Option<String>,
    pub inference_count: Option<String>,
    pub execution_count: Option<String>,
    pub batch_stats: Option<Vec<BatchStats>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricsResponse {
    pub model_stats: Vec<ModelStats>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub processing_time_ms: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub processing_time_ms: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageClassificationResponse {
    pub probabilities: Vec<Vec<f32>>,
    pub processing_time_ms: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResponse {
    pub text_embeddings: Option<Vec<Vec<f32>>>,
    pub image_embeddings: Option<Vec<Vec<f32>>>,
    pub processing_time_ms: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelClassificationLabelsResponse {
    pub labels: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEmbeddingDimsResponse {
    pub embedding_size: u64,
}
