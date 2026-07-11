use ed25519_dalek::VerifyingKey;

use super::token::{parse_public_key_base64, verify_unlock_token};
use crate::copilot::v2::inference::CONTEXT_LENGTH;

pub const DEGRADED_CONTEXT_LENGTH: usize = 32;

/// Decides the effective context window from an unlock token.
///
/// The Ed25519 public key is baked in at compile time via
/// `THYLLORE_UNLOCK_PUBKEY_B64` (base64 of the 32 raw key bytes). Without a
/// baked-in key no token can unlock, so dev builds always degrade to
/// [`DEGRADED_CONTEXT_LENGTH`].
pub struct UnlockGate {
    verifying_key: Option<VerifyingKey>,
}

impl UnlockGate {
    pub fn from_build_env() -> Self {
        const PUBKEY_B64: &str = match option_env!("THYLLORE_UNLOCK_PUBKEY_B64") {
            Some(value) => value,
            None => "",
        };

        if PUBKEY_B64.is_empty() {
            return Self {
                verifying_key: None,
            };
        }
        Self {
            verifying_key: parse_public_key_base64(obfstr::obfstr!(PUBKEY_B64)),
        }
    }

    pub fn with_verifying_key(verifying_key: VerifyingKey) -> Self {
        Self {
            verifying_key: Some(verifying_key),
        }
    }

    pub fn effective_context_length(&self, unlock_token: Option<&str>, now_unix: u64) -> usize {
        match (&self.verifying_key, unlock_token) {
            (Some(key), Some(token)) if verify_unlock_token(token, key, now_unix) => CONTEXT_LENGTH,
            _ => DEGRADED_CONTEXT_LENGTH,
        }
    }
}

/// Degrades a full context window in place: only the most recent
/// [`DEGRADED_CONTEXT_LENGTH`] samples are kept, older samples are zero-padded
/// so the input length stays [`CONTEXT_LENGTH`].
pub fn degrade_context_window(context: &mut [f32]) {
    let keep_from = context.len().saturating_sub(DEGRADED_CONTEXT_LENGTH);
    for value in &mut context[..keep_from] {
        *value = 0.0;
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unlock::token::test_support::{signing_key, token_with_exp};

    const NOW: u64 = 1_752_200_000;

    fn gate() -> UnlockGate {
        UnlockGate::with_verifying_key(signing_key().verifying_key())
    }

    #[test]
    fn valid_token_unlocks_full_context() {
        let token = token_with_exp(&signing_key(), NOW + 3600);
        assert_eq!(
            gate().effective_context_length(Some(&token), NOW),
            CONTEXT_LENGTH
        );
    }

    #[test]
    fn expired_token_degrades() {
        let token = token_with_exp(&signing_key(), NOW - 1);
        assert_eq!(
            gate().effective_context_length(Some(&token), NOW),
            DEGRADED_CONTEXT_LENGTH
        );
    }

    #[test]
    fn missing_token_degrades() {
        assert_eq!(
            gate().effective_context_length(None, NOW),
            DEGRADED_CONTEXT_LENGTH
        );
    }

    #[test]
    fn garbage_token_degrades() {
        assert_eq!(
            gate().effective_context_length(Some("garbage"), NOW),
            DEGRADED_CONTEXT_LENGTH
        );
    }

    #[test]
    fn from_build_env_key_presence_matches_build_env() {
        let gate = UnlockGate::from_build_env();
        match option_env!("THYLLORE_UNLOCK_PUBKEY_B64") {
            Some(_) => assert!(gate.verifying_key.is_some()),
            None => assert!(gate.verifying_key.is_none()),
        }
    }

    #[test]
    fn gate_without_key_never_unlocks() {
        let keyless = UnlockGate {
            verifying_key: None,
        };
        let token = token_with_exp(&signing_key(), NOW + 3600);
        assert_eq!(
            keyless.effective_context_length(Some(&token), NOW),
            DEGRADED_CONTEXT_LENGTH
        );
    }

    #[test]
    fn degrade_zero_pads_older_half_and_keeps_recent() {
        let mut context: Vec<f32> = (1..=CONTEXT_LENGTH).map(|v| v as f32).collect();
        degrade_context_window(&mut context);

        let pad = CONTEXT_LENGTH - DEGRADED_CONTEXT_LENGTH;
        assert!(context[..pad].iter().all(|&v| v == 0.0));
        for (i, &value) in context[pad..].iter().enumerate() {
            assert_eq!(value, (pad + i + 1) as f32);
        }
    }

    #[test]
    fn degrade_on_short_window_is_noop() {
        let mut context = vec![1.0_f32; DEGRADED_CONTEXT_LENGTH];
        degrade_context_window(&mut context);
        assert!(context.iter().all(|&v| v == 1.0));
    }
}
