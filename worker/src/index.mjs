/**
 * Curve Copilot feedback ingest (Cloudflare Worker + R2).
 *
 * POST /v1/feedback  gzip JSONL batch of anonymized correction pairs.
 *                    Valid (even empty) batches receive a short-lived Ed25519
 *                    unlock_token so mode B clients keep ctx64 unlocked.
 * POST /v1/message   one free-text feedback message (separate storage prefix).
 * POST /v1/license/refresh  reserved for mode C (seat-bound licensing), 501.
 *
 * Secrets: INGEST_TOKEN, UNLOCK_PRIVATE_KEY_PKCS8_B64 (see wrangler.toml).
 * Hard rate limiting is expected from Cloudflare Rate Limiting Rules in front
 * of this Worker; handlers only enforce size and schema.
 */

const FEEDBACK_SCHEMA = "curve_copilot_feedback/v0";
const REQUIRED_RECORD_KEYS = ["schema", "channel", "fps", "context", "prediction"];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method !== "POST") {
      return new Response("method not allowed", { status: 405 });
    }
    if (!isAuthorized(request, env)) {
      return new Response("unauthorized", { status: 401 });
    }

    switch (url.pathname) {
      case "/v1/feedback":
        return handleFeedback(request, env);
      case "/v1/message":
        return handleMessage(request, env);
      case "/v1/license/refresh":
        return new Response("license refresh not implemented yet", { status: 501 });
      default:
        return new Response("not found", { status: 404 });
    }
  },
};

function isAuthorized(request, env) {
  const header = request.headers.get("Authorization") ?? "";
  return header === `Bearer ${env.INGEST_TOKEN}`;
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
  const { token, exp } = await signUnlockToken(env, ttl);
  return Response.json({ unlock_token: token, exp });
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

async function signUnlockToken(env, ttlSeconds) {
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const payload = new TextEncoder().encode(JSON.stringify({ exp }));

  const der = base64Decode(env.UNLOCK_PRIVATE_KEY_PKCS8_B64);
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
