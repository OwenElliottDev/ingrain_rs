# `ingrain-rs`

A Rust client for interacting with Ingrain Inference.

## Quick Start 

```rust
use ingrain_rs::IngrainClient;
use ingrain_rs::models::ModelLibrary;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client
    let client = IngrainClient::new("http://localhost:8687", "http://localhost:8686");
    let model_health = client.model_server_health().await?;
    println!("Model server: {}", model_health.message);
    let inference_health = client.inference_server_health().await?;
    println!("Inference server: {}", inference_health.message);

    // Load a model
    let model_id = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K".to_string();
    client.load_model(model_id.clone(), ModelLibrary::OpenClip).await?;
    println!("Model loaded.");

    // Perform inference and get embeddings
    let texts = Some(vec!["Text 1".to_string(), "Text 2".to_string()]);
    let images = Some(vec!["data:image/jpeg;base64,...".to_string()]);

    let result = client.embed(model_id.clone(), texts, images, None, None, None).await?;
    
    if let Some(text_embeddings) = &result.text_embeddings {
        println!("Text Embedding 0 length: {}", text_embeddings[0].len());
    }

    if let Some(image_embeddings) = &result.image_embeddings {
        println!("Image Embedding 0 length: {}", image_embeddings[0].len());
    }

    // Unload model
    client.unload_model(model_id).await?;
    println!("Model unloaded.");

    Ok(())
}

```

