use std::error::Error;
use std::time::Duration;

use reqwest::RequestBuilder;
use serde::de::DeserializeOwned;
use tokio::time::sleep;

pub async fn retry<T>(
    request_builder: RequestBuilder,
    retries: u16,
    retry_delay_ms: u64,
) -> Result<T, Box<dyn Error>>
where
    T: DeserializeOwned + Send + 'static,
{
    let mut last_err: Option<String> = None;

    for attempt in 0..retries + 1 {
        let request = request_builder
            .try_clone()
            .ok_or("Failed to clone request")?;

        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let body = response.text().await?;

                if status.is_success() {
                    match serde_json::from_str::<T>(&body) {
                        Ok(parsed) => return Ok(parsed),
                        Err(e) => {
                            last_err =
                                Some(format!("Failed to parse response: {} (body: {})", e, body));
                        }
                    }
                } else {
                    last_err = Some(format!(
                        "Request failed with status: {} (body: {})",
                        status, body
                    ));
                }
            }
            Err(e) => {
                last_err = Some(format!("Network error: {}", e));
            }
        }

        if attempt < retries {
            sleep(Duration::from_millis(retry_delay_ms)).await;
        }
    }

    Err(last_err
        .unwrap_or_else(|| "Unknown error".to_string())
        .into())
}
