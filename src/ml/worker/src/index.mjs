/**
 * Curve Copilot feedback ingest + mode C licensing (Cloudflare Worker + R2 + DO).
 *
 * POST /v1/feedback  gzip JSONL batch of anonymized correction pairs.
 *                    Valid (even empty) batches receive a short-lived Ed25519
 *                    full_token so mode B clients keep ctx64.
 * POST /v1/message   one free-text feedback message (separate storage prefix).
 * POST /v1/license/refresh    mode C: {license_key, device_id} -> full_token
 *                    while the LicenseSeats Durable Object grants a seat;
 *                    copied license keys exhaust seats and are refused (403).
 * POST /v1/license/provision  admin-only upsert of a license's max_seats/status.
 *
 * Secrets: INGEST_TOKEN, ADMIN_TOKEN, FULL_TOKEN_PRIVATE_KEY_PKCS8_B64
 * (see wrangler.toml). Hard rate limiting is expected from Cloudflare Rate
 * Limiting Rules in front of this Worker; handlers only enforce size and schema.
 */

const FEEDBACK_SCHEMA = "curve_copilot_feedback/v1";
const REQUIRED_RECORD_KEYS = ["schema", "channel", "fps", "context", "prediction"];
const MAX_KEY_STRING_LENGTH = 128;
const MAX_SEATS_LIMIT = 100;
const LICENSE_STATUSES = ["active", "revoked"];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method !== "POST") {
      return new Response("method not allowed", { status: 405 });
    }

    switch (url.pathname) {
      case "/v1/feedback":
        if (!isBearerAuthorized(request, env.INGEST_TOKEN)) {
          return new Response("unauthorized", { status: 401 });
        }
        return handleFeedback(request, env);
      case "/v1/message":
        if (!isBearerAuthorized(request, env.INGEST_TOKEN)) {
          return new Response("unauthorized", { status: 401 });
        }
        return handleMessage(request, env);
      case "/v1/license/refresh":
        return handleLicenseRefresh(request, env);
      case "/v1/license/provision":
        if (!isBearerAuthorized(request, env.ADMIN_TOKEN)) {
          return new Response("unauthorized", { status: 401 });
        }
        return handleLicenseProvision(request, env);
      default:
        if (!isBearerAuthorized(request, env.INGEST_TOKEN)) {
          return new Response("unauthorized", { status: 401 });
        }
        return new Response("not found", { status: 404 });
    }
  },
};

function isBearerAuthorized(request, expectedToken) {
  const header = request.headers.get("Authorization") ?? "";
  return typeof expectedToken === "string"
    && expectedToken.length > 0
    && header === `Bearer ${expectedToken}`;
}

function isValidKeyString(value) {
  return typeof value === "string" && value.length > 0 && value.length <= MAX_KEY_STRING_LENGTH;
}

async function readJsonBody(request, env) {
  const body = await readBodyWithLimit(request, Number(env.MAX_BODY_BYTES));
  if (body === null) {
    return { error: new Response("body too large", { status: 413 }) };
  }
  try {
    return { payload: JSON.parse(new TextDecoder().decode(body)) };
  } catch {
    return { error: new Response("invalid json", { status: 400 }) };
  }
}

async function handleLicenseRefresh(request, env) {
  const { payload, error } = await readJsonBody(request, env);
  if (error) {
    return error;
  }
  if (!isValidKeyString(payload.license_key) || !isValidKeyString(payload.device_id)) {
    return new Response("missing license_key or device_id", { status: 400 });
  }

  const stub = env.LICENSE_SEATS.get(env.LICENSE_SEATS.idFromName(payload.license_key));
  const decision = await stub
    .fetch("https://license-seats/refresh", {
      method: "POST",
      body: JSON.stringify({ device_id: payload.device_id }),
    })
    .then((response) => response.json());

  if (!decision.allowed) {
    return Response.json({ error: decision.reason }, { status: 403 });
  }
  const ttl = Number(env.TOKEN_TTL_SECONDS);
  const { token, exp } = await signFullToken(env, ttl);
  return Response.json({ full_token: token, exp });
}

async function handleLicenseProvision(request, env) {
  const { payload, error } = await readJsonBody(request, env);
  if (error) {
    return error;
  }
  const maxSeats = Number(payload.max_seats);
  const validSeats = Number.isInteger(maxSeats) && maxSeats >= 1 && maxSeats <= MAX_SEATS_LIMIT;
  if (!isValidKeyString(payload.license_key) || !validSeats
    || !LICENSE_STATUSES.includes(payload.status)) {
    return new Response("invalid license_key, max_seats or status", { status: 400 });
  }

  const stub = env.LICENSE_SEATS.get(env.LICENSE_SEATS.idFromName(payload.license_key));
  await stub.fetch("https://license-seats/provision", {
    method: "POST",
    body: JSON.stringify({ max_seats: maxSeats, status: payload.status }),
  });
  return new Response(null, { status: 204 });
}

/**
 * Per-license seat registry. One object per license_key (idFromName), so seat
 * checks are strictly serialized: a copied license key can never register more
 * than max_seats devices, no matter how concurrent the refresh calls are.
 *
 * Storage layout:
 *   "license" -> { max_seats, status }         (written by /provision, upsert)
 *   "seats"   -> { [device_id]: last_seen_unix } (pruned by SEAT_TTL_SECONDS)
 */
export class LicenseSeats {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);
    const payload = await request.json();
    switch (url.pathname) {
      case "/provision":
        return this.provision(payload);
      case "/refresh":
        return this.refresh(payload);
      default:
        return new Response("not found", { status: 404 });
    }
  }

  async provision(payload) {
    await this.ctx.storage.put("license", {
      max_seats: payload.max_seats,
      status: payload.status,
    });
    return Response.json({ ok: true });
  }

  async refresh(payload) {
    const license = await this.ctx.storage.get("license");
    if (license === undefined) {
      return Response.json({ allowed: false, reason: "unknown_license" });
    }
    if (license.status !== "active") {
      return Response.json({ allowed: false, reason: "revoked" });
    }

    const now = Math.floor(Date.now() / 1000);
    const seatTtl = Number(this.env.SEAT_TTL_SECONDS);
    const storedSeats = (await this.ctx.storage.get("seats")) ?? {};
    const seats = Object.fromEntries(
      Object.entries(storedSeats).filter(([, lastSeen]) => now - lastSeen <= seatTtl)
    );

    const hasSeat = payload.device_id in seats;
    if (!hasSeat && Object.keys(seats).length >= license.max_seats) {
      await this.ctx.storage.put("seats", seats);
      return Response.json({ allowed: false, reason: "seat_exhausted" });
    }
    seats[payload.device_id] = now;
    await this.ctx.storage.put("seats", seats);
    return Response.json({ allowed: true });
  }
}

async function handleFeedback(request, env) {
  const body = await readBodyWithLimit(request, Number(env.MAX_BODY_BYTES));
  if (body === null) {
    return new Response("body too large", { status: 413 });
  }

  let text;
  try {
    text = await decodeBody(body, request.headers.get("Content-Encoding"));
  } catch {
    return new Response("bad encoding", { status: 400 });
  }

  const records = parseRecords(text);
  if (records === null) {
    return new Response("invalid records", { status: 400 });
  }

  if (records.length > 0) {
    const key = feedbackObjectKey();
    await env.FEEDBACK_BUCKET.put(key, text);
  }

  const ttl = Number(env.TOKEN_TTL_SECONDS);
  const { token, exp } = await signFullToken(env, ttl);
  return Response.json({ full_token: token, exp });
}

async function handleMessage(request, env) {
  const body = await readBodyWithLimit(request, Number(env.MAX_BODY_BYTES));
  if (body === null) {
    return new Response("body too large", { status: 413 });
  }

  let message;
  try {
    message = JSON.parse(new TextDecoder().decode(body));
  } catch {
    return new Response("invalid json", { status: 400 });
  }
  if (typeof message.text !== "string" || message.text.length === 0) {
    return new Response("missing text", { status: 400 });
  }
  if (message.text.length > Number(env.MAX_MESSAGE_TEXT_BYTES)) {
    return new Response("text too long", { status: 413 });
  }

  const day = new Date().toISOString().slice(0, 10).replaceAll("-", "");
  const key = `feedback-messages/${day}/${crypto.randomUUID()}.json`;
  await env.FEEDBACK_BUCKET.put(
    key,
    JSON.stringify({
      text: message.text,
      addon_version: String(message.addon_version ?? ""),
      anon_id: String(message.anon_id ?? ""),
      ts: Number(message.ts ?? Math.floor(Date.now() / 1000)),
    })
  );
  return new Response(null, { status: 204 });
}

async function readBodyWithLimit(request, maxBytes) {
  const buffer = await request.arrayBuffer();
  if (buffer.byteLength > maxBytes) {
    return null;
  }
  return buffer;
}

async function decodeBody(buffer, contentEncoding) {
  if (contentEncoding !== "gzip") {
    return new TextDecoder().decode(buffer);
  }
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).text();
}

function parseRecords(text) {
  const lines = text.split("\n").filter((line) => line.trim().length > 0);
  const records = [];
  for (const line of lines) {
    let record;
    try {
      record = JSON.parse(line);
    } catch {
      return null;
    }
    if (record.schema !== FEEDBACK_SCHEMA) {
      return null;
    }
    if (!REQUIRED_RECORD_KEYS.every((key) => key in record)) {
      return null;
    }
    records.push(record);
  }
  return records;
}

function feedbackObjectKey() {
  const day = new Date().toISOString().slice(0, 10).replaceAll("-", "");
  return `feedback/${day}/${FEEDBACK_SCHEMA.replaceAll("/", "_")}/${crypto.randomUUID()}.jsonl`;
}

async function signFullToken(env, ttlSeconds) {
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const payload = new TextEncoder().encode(JSON.stringify({ exp }));

  const der = base64Decode(env.FULL_TOKEN_PRIVATE_KEY_PKCS8_B64);
  const key = await crypto.subtle.importKey("pkcs8", der, { name: "Ed25519" }, false, ["sign"]);
  const signature = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, key, payload));

  const token = `${base64UrlEncode(payload)}.${base64UrlEncode(signature)}`;
  return { token, exp };
}

function base64Decode(encoded) {
  const binary = atob(encoded.trim());
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

function base64UrlEncode(bytes) {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}
