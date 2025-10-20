use once_cell::sync::Lazy;
use std::collections::HashMap;
use tokio;
use tokio::sync::Mutex;

use ingrain_rs::IngrainClient;
use ingrain_rs::models::ModelLibrary;

const INFERENCE_BASE_URL: &str = "http://localhost:8686";
const MODEL_BASE_URL: &str = "http://localhost:8687";

const SENTENCE_TRANSFORMER_MODEL: &str = "intfloat/e5-small-v2";
const OPENCLIP_MODEL: &str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K";
const TIMM_MODEL: &str = "hf_hub:timm/mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k";

static MODEL_LOCKS: Lazy<HashMap<&'static str, Mutex<()>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("intfloat/e5-small-v2", Mutex::new(()));
    map.insert(
        "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        Mutex::new(()),
    );
    map.insert(
        "hf_hub:timm/mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        Mutex::new(()),
    );
    map
});

async fn with_model_lock<F, Fut>(model_name: &'static str, f: F)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let lock = MODEL_LOCKS.get(model_name).expect("Model lock not found");
    let _guard = lock.lock().await;
    f().await;
}

#[tokio::test]
async fn test_model_server_health() {
    let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);

    let res = client.model_server_health().await;

    assert!(res.is_ok());
    let resp = res.unwrap();
    assert!(resp.message != "".to_string());
}

#[tokio::test]
async fn test_inference_server_health() {
    let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);

    let res = client.inference_server_health().await;

    assert!(res.is_ok());
    let resp = res.unwrap();
    assert!(resp.message != "".to_string());
}

#[tokio::test]
async fn test_load_openclip_model() {
    with_model_lock(OPENCLIP_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client
            .load_model(OPENCLIP_MODEL.to_string(), ModelLibrary::OpenClip)
            .await;
        assert!(res.is_ok());
        let resp = res.unwrap();
        assert!(resp.message.contains("loaded"));

        let _ = client.unload_model(OPENCLIP_MODEL.to_string()).await;
    })
    .await;
}

#[tokio::test]
async fn test_load_sentence_transformers_model() {
    with_model_lock(SENTENCE_TRANSFORMER_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client
            .load_model(
                SENTENCE_TRANSFORMER_MODEL.to_string(),
                ModelLibrary::SentenceTransformers,
            )
            .await;
        assert!(res.is_ok());
        let resp = res.unwrap();
        assert!(resp.message.contains("loaded"));

        let _ = client
            .unload_model(SENTENCE_TRANSFORMER_MODEL.to_string())
            .await;
    })
    .await;
}

#[tokio::test]
async fn test_load_timm_model() {
    with_model_lock(TIMM_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client
            .load_model(TIMM_MODEL.to_string(), ModelLibrary::Timm)
            .await;
        assert!(res.is_ok());
        let resp = res.unwrap();
        assert!(resp.message.contains("loaded"));

        let _ = client.unload_model(TIMM_MODEL.to_string()).await;
    })
    .await;
}

#[tokio::test]
async fn test_embed_text() {
    with_model_lock(SENTENCE_TRANSFORMER_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client
            .load_model(
                SENTENCE_TRANSFORMER_MODEL.to_string(),
                ModelLibrary::SentenceTransformers,
            )
            .await;
        assert!(res.is_ok());

        let test_text = vec!["This is a sentence.".to_string()];
        let res = client
            .embed_text(
                SENTENCE_TRANSFORMER_MODEL.to_string(),
                test_text,
                None,
                None,
            )
            .await;

        assert!(res.is_ok());

        let resp = res.unwrap();

        assert_eq!(resp.embeddings[0].len(), 384);

        let _ = client
            .unload_model(SENTENCE_TRANSFORMER_MODEL.to_string())
            .await;
    })
    .await;
}

#[tokio::test]
async fn test_embed_image() {
    with_model_lock(OPENCLIP_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client.load_model(OPENCLIP_MODEL.to_string(), ModelLibrary::OpenClip).await;
        assert!(res.is_ok());
        let test_image = vec!["data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII=".to_string()];
        let res = client.embed_image(
            OPENCLIP_MODEL.to_string(),
            test_image,
            None,
            None,
            None
        ).await;

        assert!(res.is_ok());

        let resp = res.unwrap();
        assert_eq!(resp.embeddings[0].len(), 512);

        let _ = client.unload_model(OPENCLIP_MODEL.to_string()).await;
    }).await;
}

#[tokio::test]
async fn test_classify_image() {
    with_model_lock(TIMM_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client.load_model(TIMM_MODEL.to_string(), ModelLibrary::Timm).await;
        assert!(res.is_ok());
        let test_image = vec!["data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII=".to_string()];
        let res = client.classify_image(
            TIMM_MODEL.to_string(),
            test_image,
            None,
        ).await;

        assert!(res.is_ok());

        let resp = res.unwrap();
        assert_eq!(resp.probabilities[0].len(), 1000);

        let _ = client.unload_model(TIMM_MODEL.to_string()).await;
    }).await;
}

#[tokio::test]
async fn test_embed_text_image() {
    with_model_lock(OPENCLIP_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client.load_model(OPENCLIP_MODEL.to_string(), ModelLibrary::OpenClip).await;
        assert!(res.is_ok());
        let test_image = vec!["data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII=".to_string()];
        let test_text = vec!["Text1".to_string(), "text 2".to_string()];
        let res = client.embed(
            OPENCLIP_MODEL.to_string(),
            Some(test_text),
            Some(test_image),
            None,
            None,
            None
        ).await;

        assert!(res.is_ok());

        let resp = res.unwrap();
        assert!(resp.text_embeddings.is_some());
        assert!(resp.image_embeddings.is_some());
        let te = resp.text_embeddings.unwrap();
        assert_eq!(te.len(), 2);
        assert_eq!(te[0].len(), 512);
        assert_eq!(resp.image_embeddings.unwrap()[0].len(), 512);

        let _ = client.unload_model(OPENCLIP_MODEL.to_string()).await;
    }).await;
}

#[tokio::test]
async fn test_model_embed_dims() {
    with_model_lock(SENTENCE_TRANSFORMER_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client
            .load_model(
                SENTENCE_TRANSFORMER_MODEL.to_string(),
                ModelLibrary::SentenceTransformers,
            )
            .await;
        assert!(res.is_ok());

        let res = client
            .model_embedding_size(SENTENCE_TRANSFORMER_MODEL.to_string())
            .await;
        assert!(res.is_ok());
        let resp = res.unwrap();
        assert_eq!(resp.embedding_size, 384);
        let _ = client
            .unload_model(SENTENCE_TRANSFORMER_MODEL.to_string())
            .await;
    })
    .await;
}

#[tokio::test]
async fn test_model_labels() {
    with_model_lock(TIMM_MODEL, || async {
        let client = IngrainClient::new(MODEL_BASE_URL, INFERENCE_BASE_URL);
        let res = client
            .load_model(TIMM_MODEL.to_string(), ModelLibrary::Timm)
            .await;
        assert!(res.is_ok());

        let res = client
            .model_classification_labels(TIMM_MODEL.to_string())
            .await;
        assert!(res.is_ok());
        let resp = res.unwrap();
        assert_eq!(resp.labels.len(), 1000);
        let _ = client.unload_model(TIMM_MODEL.to_string()).await;
    })
    .await;
}
