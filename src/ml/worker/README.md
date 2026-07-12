# Curve Copilot Feedback Worker

Cloudflare Worker + R2 ingest for anonymized Curve Copilot correction pairs
(mode B builds) and free-text feedback messages. Design:
`${DocumentPath}/Rust_Rendering/Design/20260711_curve_copilot_data_collection/`.

Every Cloudflare interaction is done with **curl + jq only** — no npm, no
wrangler, no Node.js. The worker code (`src/index.mjs`) has zero imports and
uses only Workers runtime built-ins, so there is nothing to `npm install`.

## Endpoints

| Route | Purpose | Response |
|---|---|---|
| `POST /v1/feedback` | gzip JSONL batch (`curve_copilot_feedback/v0`); empty batch = token handshake | `{unlock_token, exp}` |
| `POST /v1/message` | one free-text message (4KB limit) | `204` |
| `POST /v1/license/refresh` | mode C seat licensing: `{license_key, device_id}`; the LicenseSeats Durable Object grants or refuses a seat | `{unlock_token, exp}` or `403 {error}` |
| `POST /v1/license/provision` | admin upsert of a license: `{license_key, max_seats, status}` (`active` / `revoked`) | `204` |

`/v1/feedback` and `/v1/message` require `Authorization: Bearer <INGEST_TOKEN>`;
`/v1/license/provision` requires `Bearer <ADMIN_TOKEN>`; `/v1/license/refresh`
authenticates by the license_key itself. Seat state lives in the `LicenseSeats`
SQLite Durable Object (one object per license_key, so seat checks are strictly
serialized); seats idle for `SEAT_TTL_SECONDS` (30 days) are pruned, which lets
legitimate users move to a new machine while copied keys stay locked out.

## One-time bootstrap (dashboard, no CLI)

Done once in the Cloudflare dashboard — these steps mint the credentials the
scripts need:

1. Sign in (GitHub-linked account works) and enable R2 Object Storage.
2. Create a **User API Token** with permissions: `Account > Workers Scripts >
   Edit`, `Account > Workers R2 Storage > Edit`, `Account > Account Settings >
   Read`. Copy the token and your Account ID.

## Environments: separate test and production buckets

| Env | Worker | R2 bucket |
|---|---|---|
| prod (default) | `curve-copilot-feedback` | `curve-feedback` |
| test | `curve-copilot-feedback-test` | `curve-feedback-test` |

`deploy.sh --env test` appends a `-test` suffix to both names so staging /
test data never lands in the production bucket. `wrangler.toml` holds the
production base names.

## Deploy (curl)

`deploy.sh` reads non-secret config from `wrangler.toml` (the config source of
truth) and uploads the worker, its R2 + Durable Object bindings, plain-text
vars, and the secrets via the Cloudflare REST API. The `LicenseSeats` DO
migration is applied automatically on the first deploy of an environment (plain
upload fails with code 10061, the script retries once with the migration).
Secrets come from the environment and are never printed or written to disk
unencrypted.

```bash
# 1. Generate the signing keypair (writes secrets/, which is gitignored)
scripts/gen_license_keypair.sh

# 2. Deploy
export CF_API_TOKEN=...            # the token from bootstrap step 2
export CF_ACCOUNT_ID=...           # your account id
export THYLLORE_INGEST_TOKEN=...   # a random shared token; also bake into mode B builds
export THYLLORE_ADMIN_TOKEN=...    # a random token guarding /v1/license/provision
export THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE=secrets/private_key_pkcs8.b64
cd src/ml/worker
./deploy.sh --env test             # test bucket first; drop --env or use prod for production
./deploy.sh                        # --dry-run validates config without any API call
```

On success it prints the `https://<name>.<subdomain>.workers.dev` URL. Use its
`/v1/feedback` form as `THYLLORE_FEEDBACK_ENDPOINT` when building mode B addons.
The matching public key (`secrets/public_key.b64`) is baked into the wheel and
mode B/C builds via `THYLLORE_UNLOCK_PUBKEY_B64`.

Options: `--env prod|test`, `--dry-run` (build + validate, no API calls),
`--skip-bucket` (do not create the R2 bucket).

## Test locally through the production path (no npm)

The same code path as production is exercised locally by running the real
`src/index.mjs` in a local **workerd** instance. workerd is Cloudflare's own
Workers runtime; its standalone binary is downloaded with curl and pinned by
SHA256 (the npm `workerd` package merely wraps the same binary), so no npm is
involved.

```bash
./test_local_e2e.sh          # mode B: feedback handshake -> token -> wheel ctx64
./test_license_seat_e2e.sh   # layer 5 / mode C: seat grant, exhaustion, revocation -> wheel
```

Both scripts (shared helpers in `lib_e2e.sh`) generate an Ed25519 keypair,
build the wheel with its public key baked in, start local workerd, and assert
the full chain:

```
local workerd (real index.mjs + LicenseSeats DO)  --Ed25519 unlock_token-->  wheel -> ctx64
```

The seat e2e additionally proves copy invalidation: a third device on a
2-seat license is refused (`seat_exhausted`) and stays at ctx32. The DO runs
with in-memory storage locally, so every run starts from a clean seat table.
To just serve the worker locally (e.g. to point the addon at it):

```bash
INGEST_TOKEN=... ADMIN_TOKEN=... \
UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE=secrets/private_key_pkcs8.b64 \
./run_local.sh --port 8787
```

R2 is not bound locally (workerd's R2 binding needs the miniflare/npm
simulator), so the local run covers everything except the R2 write. The R2
write path is verified against the **`curve-feedback-test`** bucket via a test
deployment (`deploy.sh --env test`) — this is why the buckets are separated.

## Verify a deployed worker (curl)

`smoke.sh` exercises a deployed URL: unauthorized request rejected (401),
authorized message accepted (204, R2 write), empty feedback batch returns an
`unlock_token`. Use `--skip-r2` for a local run without R2.

```bash
WORKER_URL=https://<name>-test.<subdomain>.workers.dev \
THYLLORE_INGEST_TOKEN=... ./smoke.sh
```

`license_smoke.sh` walks the seat lifecycle on a deployed worker with a
single-use random license (grant / re-grant / exhaustion / revocation) and
leaves it revoked. Right after a deploy, wait ~20s for propagation — hitting
the previous script version returns 401 on `/v1/license/provision`.

```bash
WORKER_URL=https://<name>-test.<subdomain>.workers.dev \
THYLLORE_ADMIN_TOKEN=... ./license_smoke.sh
```

Uploads are also validated server-side: an invalid worker makes `deploy.sh`
fail with the Cloudflare compile error.

## Storage layout (R2 `curve-feedback`)

```
feedback/<yyyymmdd>/curve_copilot_feedback_v0/<uuid>.jsonl
feedback-messages/<yyyymmdd>/<uuid>.json
```

## Rate limiting

Handlers enforce size and schema only. Configure Cloudflare Rate Limiting
Rules in front of the Worker (per-IP) in the dashboard.

## Training-side ingest (separate repo)

Reading these objects for training lives in `../AnimationModelTraining`, not
here. Keep it npm-free too: R2 object GET/LIST is the S3-compatible API, so use
curl with AWS SigV4, or a standalone binary such as `rclone` / the `aws` CLI
pointed at the R2 endpoint with an R2 access key — never wrangler/npm. Delete or
archive consumed objects after ingest.
