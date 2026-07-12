use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use ed25519_dalek::{Signature, VerifyingKey};

pub fn parse_public_key_base64(encoded: &str) -> Option<VerifyingKey> {
    let bytes = STANDARD.decode(encoded.trim()).ok()?;
    let bytes: [u8; 32] = bytes.try_into().ok()?;
    VerifyingKey::from_bytes(&bytes).ok()
}

/// Token format: `base64url(payload_json).base64url(ed25519_signature)`,
/// payload is `{"exp": <unix seconds>}` signed over the raw payload bytes.
pub fn verify_token(token: &str, key: &VerifyingKey, now_unix: u64) -> bool {
    let Some((payload_b64, signature_b64)) = token.split_once('.') else {
        return false;
    };
    let Ok(payload) = URL_SAFE_NO_PAD.decode(payload_b64) else {
        return false;
    };
    let Ok(signature_bytes) = URL_SAFE_NO_PAD.decode(signature_b64) else {
        return false;
    };
    let Ok(signature_bytes) = <[u8; 64]>::try_from(signature_bytes) else {
        return false;
    };

    let signature = Signature::from_bytes(&signature_bytes);
    if key.verify_strict(&payload, &signature).is_err() {
        return false;
    }

    let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&payload) else {
        return false;
    };
    let Some(exp) = claims.get(obfstr::obfstr!("exp")).and_then(|v| v.as_u64()) else {
        return false;
    };
    exp > now_unix
}

#[cfg(test)]
pub mod test_support {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use ed25519_dalek::{Signer, SigningKey};

    pub fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    pub fn sign_token(key: &SigningKey, payload: &str) -> String {
        let signature = key.sign(payload.as_bytes());
        format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(payload.as_bytes()),
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        )
    }

    pub fn token_with_exp(key: &SigningKey, exp: u64) -> String {
        sign_token(key, &format!("{{\"exp\":{exp}}}"))
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{sign_token, signing_key, token_with_exp};
    use super::*;
    use ed25519_dalek::SigningKey;

    const NOW: u64 = 1_752_200_000;

    #[test]
    fn valid_future_token_verifies() {
        let key = signing_key();
        let token = token_with_exp(&key, NOW + 3600);
        assert!(verify_token(&token, &key.verifying_key(), NOW));
    }

    #[test]
    fn expired_token_is_rejected() {
        let key = signing_key();
        let token = token_with_exp(&key, NOW - 1);
        assert!(!verify_token(&token, &key.verifying_key(), NOW));
    }

    #[test]
    fn exp_equal_to_now_is_rejected() {
        let key = signing_key();
        let token = token_with_exp(&key, NOW);
        assert!(!verify_token(&token, &key.verifying_key(), NOW));
    }

    #[test]
    fn token_signed_by_other_key_is_rejected() {
        let key = signing_key();
        let other = SigningKey::from_bytes(&[9u8; 32]);
        let token = token_with_exp(&other, NOW + 3600);
        assert!(!verify_token(&token, &key.verifying_key(), NOW));
    }

    #[test]
    fn tampered_payload_is_rejected() {
        let key = signing_key();
        let token = token_with_exp(&key, NOW + 3600);
        let (_, signature) = token.split_once('.').unwrap();
        let forged_payload = URL_SAFE_NO_PAD.encode(format!("{{\"exp\":{}}}", NOW + 999_999));
        let forged = format!("{forged_payload}.{signature}");
        assert!(!verify_token(&forged, &key.verifying_key(), NOW));
    }

    #[test]
    fn payload_without_exp_is_rejected() {
        let key = signing_key();
        let token = sign_token(&key, "{\"mode\":\"B\"}");
        assert!(!verify_token(&token, &key.verifying_key(), NOW));
    }

    #[test]
    fn malformed_tokens_are_rejected() {
        let key = signing_key();
        let verifying = key.verifying_key();
        for token in ["", "no-dot", "a.b", "!!.!!", "eyJleHAiOjF9."] {
            assert!(
                !verify_token(token, &verifying, NOW),
                "token {token:?} must not verify"
            );
        }
    }

    #[test]
    fn public_key_base64_round_trips() {
        let key = signing_key();
        let encoded = STANDARD.encode(key.verifying_key().to_bytes());
        let parsed = parse_public_key_base64(&encoded).expect("parse");
        assert_eq!(parsed.to_bytes(), key.verifying_key().to_bytes());
    }

    #[test]
    fn invalid_public_key_base64_is_none() {
        assert!(parse_public_key_base64("not-base64!").is_none());
        assert!(parse_public_key_base64(&STANDARD.encode([0u8; 16])).is_none());
    }
}
