using Workerd = import "/workerd/workerd.capnp";

# Runs the real src/ml/worker/src/index.mjs in a local workerd instance so the same
# code path as production (gzip decode, schema validation, Ed25519 signing,
# response shape) can be exercised without npm or wrangler. Secrets are read
# from the process environment at startup and never stored in this file.
#
# Started by run_local.sh; the HTTP socket address is substituted there.

const config :Workerd.Config = (
  services = [ (name = "main", worker = .mainWorker) ],
  sockets = [ (name = "http", address = "SOCKET_ADDRESS", http = (), service = "main") ],
);

const mainWorker :Workerd.Worker = (
  modules = [ (name = "index.mjs", esModule = embed "src/index.mjs") ],
  compatibilityDate = "2026-07-01",
  durableObjectNamespaces = [
    (className = "LicenseSeats", uniqueKey = "thyllore-license-seats"),
  ],
  durableObjectStorage = (inMemory = void),
  bindings = [
    (name = "INGEST_TOKEN", fromEnvironment = "INGEST_TOKEN"),
    (name = "ADMIN_TOKEN", fromEnvironment = "ADMIN_TOKEN"),
    (name = "UNLOCK_PRIVATE_KEY_PKCS8_B64", fromEnvironment = "UNLOCK_PRIVATE_KEY_PKCS8_B64"),
    (name = "TOKEN_TTL_SECONDS", text = "604800"),
    (name = "MAX_BODY_BYTES", text = "1048576"),
    (name = "MAX_MESSAGE_TEXT_BYTES", text = "4096"),
    (name = "SEAT_TTL_SECONDS", text = "2592000"),
    (name = "LICENSE_SEATS", durableObjectNamespace = "LicenseSeats"),
  ],
);
