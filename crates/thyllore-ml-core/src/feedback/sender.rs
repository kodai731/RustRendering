use std::io::Write;
use std::time::Duration;

use anyhow::Context;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Serialize;

use super::record::FEEDBACK_SCHEMA_VERSION;

pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
pub const USER_AGENT: &str = "ThylloreCurveCopilot/1.0";

pub struct FeedbackEndpoint {
    pub url: String,
    pub ingest_token: String,
    pub anon_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedbackResponse {
    pub full_token: Option<String>,
    pub exp: Option<u64>,
}

pub fn message_endpoint_from_feedback(feedback_url: &str) -> String {
    match feedback_url.rsplit_once('/') {
        Some((base, _)) => format!("{base}/message"),
        None => format!("{feedback_url}/message"),
    }
}

/// POSTs one gzip JSONL batch to `/v1/feedback`. An empty batch is a valid
/// token-refresh handshake. The caller decides what to do with the returned
/// `full_token` (cache it, verify it with `DegradeGate`, or just log it).
pub fn send_feedback_batch<T: Serialize>(
    endpoint: &FeedbackEndpoint,
    records: &[T],
) -> anyhow::Result<FeedbackResponse> {
    let lines = records
        .iter()
        .map(|record| serde_json::to_string(record).context("failed to serialize record"))
        .collect::<anyhow::Result<Vec<_>>>()?
        .join("\n");

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(lines.as_bytes())?;
    let body = encoder.finish()?;

    let response = ureq::post(&endpoint.url)
        .timeout(REQUEST_TIMEOUT)
        .set(
            "Authorization",
            &format!("Bearer {}", endpoint.ingest_token),
        )
        .set("Content-Type", "application/x-ndjson")
        .set("Content-Encoding", "gzip")
        .set("X-Schema-Version", FEEDBACK_SCHEMA_VERSION)
        .set("X-Anon-Id", &endpoint.anon_id)
        .set("User-Agent", USER_AGENT)
        .send_bytes(&body)
        .context("feedback batch POST failed")?;

    let payload: serde_json::Value = response
        .into_json()
        .context("feedback response is not JSON")?;
    Ok(FeedbackResponse {
        full_token: payload
            .get("full_token")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        exp: payload.get("exp").and_then(|value| value.as_u64()),
    })
}

/// POSTs one free-text feedback message to `/v1/message`.
pub fn send_message(
    endpoint: &FeedbackEndpoint,
    text: &str,
    addon_version: &str,
    ts: u64,
) -> anyhow::Result<()> {
    let body = serde_json::json!({
        "text": text,
        "addon_version": addon_version,
        "anon_id": endpoint.anon_id,
        "ts": ts,
    });

    ureq::post(&message_endpoint_from_feedback(&endpoint.url))
        .timeout(REQUEST_TIMEOUT)
        .set(
            "Authorization",
            &format!("Bearer {}", endpoint.ingest_token),
        )
        .set("Content-Type", "application/json")
        .set("User-Agent", USER_AGENT)
        .send_string(&body.to_string())
        .context("message POST failed")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_endpoint_replaces_last_path_segment() {
        assert_eq!(
            message_endpoint_from_feedback("https://example.com/v1/feedback"),
            "https://example.com/v1/message"
        );
    }

    #[test]
    fn send_feedback_batch_fails_on_unreachable_endpoint() {
        let endpoint = FeedbackEndpoint {
            url: "http://127.0.0.1:1/v1/feedback".to_string(),
            ingest_token: "token".to_string(),
            anon_id: "anon".to_string(),
        };
        let records: Vec<serde_json::Value> = Vec::new();
        assert!(send_feedback_batch(&endpoint, &records).is_err());
    }
}
