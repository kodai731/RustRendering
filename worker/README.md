# Curve Copilot Feedback Worker

Cloudflare Worker + R2 ingest for anonymized Curve Copilot correction pairs
(mode B builds) and free-text feedback messages. Design:
`${DocumentPath}/Rust_Rendering/Design/20260711_curve_copilot_data_collection/`.

## Endpoints

| Route | Purpose | Response |
|---|---|---|
| `POST /v1/feedback` | gzip JSONL batch (`curve_copilot_feedback/v0`); empty batch = token handshake | `{unlock_token, exp}` |
| `POST /v1/message` | one free-text message (4KB limit) | `204` |
| `POST /v1/license/refresh` | mode C seat licensing (not implemented) | `501` |

All routes require `Authorization: Bearer <INGEST_TOKEN>`.

## Deploy

```bash
npm i -g wrangler
wrangler login                                  # browser OAuth (GitHub-linked account works)
wrangler r2 bucket create curve-feedback
cd worker
wrangler secret put INGEST_TOKEN                # shared bearer token for mode B builds
wrangler secret put UNLOCK_PRIVATE_KEY_PKCS8_B64  # Ed25519 private key, base64 PKCS8 DER
wrangler deploy
```

`UNLOCK_PRIVATE_KEY_PKCS8_B64` is the PEM body (base64 lines joined, headers
stripped) of `secrets/private_key.pem` from `scripts/gen_license_keypair.sh`.
The matching public key must be baked into the wheel at build time via
`THYLLORE_UNLOCK_PUBKEY_B64` (base64 of the 32 raw public key bytes) and into
mode B/C addon builds via the same environment variable.

## Storage layout (R2 `curve-feedback`)

```
feedback/<yyyymmdd>/curve_copilot_feedback_v0/<uuid>.jsonl
feedback-messages/<yyyymmdd>/<uuid>.json
```

Training-side ingest reads these with `wrangler r2 object` (no S3 API), then
deletes/archives consumed objects.

## Rate limiting

Handlers enforce size and schema only. Configure Cloudflare Rate Limiting
Rules in front of the Worker (per-IP) in the dashboard.
