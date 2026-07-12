use crate::feedback::{REQUEST_TIMEOUT, USER_AGENT};

pub struct LicenseSession {
    pub full_token: String,
    pub exp: u64,
}

/// `discard_cached_token` is true only for explicit server refusals
/// (unknown_license / revoked / seat_exhausted); network failures keep the
/// cached token until it expires.
pub struct LicenseRefusal {
    pub reason: String,
    pub discard_cached_token: bool,
}

pub fn refresh_license(
    endpoint: &str,
    license_key: &str,
    device_id: &str,
) -> Result<LicenseSession, LicenseRefusal> {
    let body = serde_json::json!({
        "license_key": license_key,
        "device_id": device_id,
    });

    let response = ureq::post(endpoint)
        .timeout(REQUEST_TIMEOUT)
        .set("Content-Type", "application/json")
        .set("User-Agent", USER_AGENT)
        .send_string(&body.to_string());

    match response {
        Ok(response) => {
            let payload: serde_json::Value = response.into_json().map_err(|_| LicenseRefusal {
                reason: "invalid_response".to_string(),
                discard_cached_token: false,
            })?;
            parse_session(&payload).ok_or(LicenseRefusal {
                reason: "invalid_response".to_string(),
                discard_cached_token: false,
            })
        }
        Err(ureq::Error::Status(403, response)) => Err(LicenseRefusal {
            reason: refusal_reason(response),
            discard_cached_token: true,
        }),
        Err(ureq::Error::Status(code, _)) => Err(LicenseRefusal {
            reason: format!("http_{code}"),
            discard_cached_token: false,
        }),
        Err(ureq::Error::Transport(_)) => Err(LicenseRefusal {
            reason: "network_error".to_string(),
            discard_cached_token: false,
        }),
    }
}

fn parse_session(payload: &serde_json::Value) -> Option<LicenseSession> {
    let full_token = payload.get("full_token")?.as_str()?.to_string();
    let exp = payload.get("exp")?.as_u64()?;
    Some(LicenseSession { full_token, exp })
}

fn refusal_reason(response: ureq::Response) -> String {
    response
        .into_json::<serde_json::Value>()
        .ok()
        .and_then(|payload| {
            payload
                .get("error")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "refused".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_session_requires_token_and_exp() {
        let full = serde_json::json!({"full_token": "a.b", "exp": 123});
        let session = parse_session(&full).expect("session");
        assert_eq!(session.full_token, "a.b");
        assert_eq!(session.exp, 123);

        assert!(parse_session(&serde_json::json!({"full_token": "a.b"})).is_none());
        assert!(parse_session(&serde_json::json!({"exp": 123})).is_none());
        assert!(parse_session(&serde_json::json!({"full_token": 5, "exp": 123})).is_none());
    }

    #[test]
    fn unreachable_endpoint_is_network_error_and_keeps_token() {
        let refusal = refresh_license("http://127.0.0.1:1/v1/license/refresh", "key", "device")
            .err()
            .expect("refusal");
        assert_eq!(refusal.reason, "network_error");
        assert!(!refusal.discard_cached_token);
    }
}
