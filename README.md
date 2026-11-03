# Fatrocu Server 3.1

Rust terminal onboarding and llama.cpp model management for Windows and Linux. No custom /process API is exposed. The managed runtime supplies an HTTP/OpenAI-compatible text/image API.

## Build and start

Keep the fatrocu-cli repository beside this repository at the revision in `core-revision.txt`; it owns the shared utilities. Build with `cargo build --locked`, test with `cargo test --locked --all-targets`, and lint with `cargo clippy --locked --all-targets -- -D warnings`.

```text
fatrocu-server
fatrocu-server setup
fatrocu-server setup --no-start
fatrocu-server start
fatrocu-server status
```

The interactive command offers Standard and Advanced. Standard installs the verified upstream CPU runtime, the default model and its visual projector, saves a local configuration, then starts the model. Subsequent `start` reuses the files. Downloads and installation stay in application data; nothing is added to a source repository or release artifact. Use `--data-dir` or FATROCU_SERVER_DATA to choose another location. On Windows `setup --desktop-script` creates a new Start Fatrocu Model.cmd launch script without overwriting an existing script.

Default model: `unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL`. The tested repository revision is e87f176479d0855a907a41277aca2f8ee7a09523, with Qwen3.5-4B-UD-Q4_K_XL.gguf and mmproj-F16.gguf. Both are checked against their Hugging Face LFS SHA-256 hashes. The pinned upstream llama.cpp build is b12069; this is an upstream build tag, independent of Fatrocu stable release numbering. The published runtime asset SHA-256 digest is verified before safe archive extraction. The default is CPU-only, one inference slot, a 32768-token context and at least 1024 image tokens for grounding.

The default bind address is **127.0.0.1:1842**. Installation never starts a network-exposed service. The tool does not change firewall rules, register a background OS service, or configure public TLS automatically.

## Advanced models and hosting

```text
fatrocu-server setup --advanced --model <OWNER/REPO:QUANTIZATION> --revision <REVISION> --model-file <FILE.gguf> --projector-file <PROJECTOR.gguf>
fatrocu-server setup --advanced --host 0.0.0.0 --allow-network-exposure --api-key-env FATROCU_MODEL_API_KEY
fatrocu-server configure <CONFIG.json>
```

Advanced interactive setup asks for the model and bind address, and requires typing HOST before enabling network access. Non-loopback binding requires explicit authorization and a named API-key environment variable. The key is passed to llama.cpp through LLAMA_API_KEY and is never written into server.json or command-line arguments. Set it before starting. Protect any hosted endpoint with firewall restrictions and an authenticated TLS proxy; the tool explains exposure before setup. `configure` supports manually selected paths, context size and GPU layer count; its validation enforces the same network rules.

Custom model downloads resolve the requested revision to its exact repository commit. Ambiguous model/projector selection requires explicit filenames. Failed or mismatched downloads are never executed. Corrupt existing files are rejected, and interrupted temporary downloads are not treated as completed installations.

## Connection diagnosis

```text
curl http://127.0.0.1:1842/health
curl http://127.0.0.1:1842/v1/models
```

The desktop and CLI use their own HTTP client; cURL is only for manual diagnosis. Remote consumers must explicitly disclose sending document contents before inference. This server does not collect application telemetry or upload source documents to another model service.

## Validation state

Windows Standard setup was exercised with real verified runtime/model/projector downloads, local startup, health/models checks and a document-free READY request on 2026-11-03. No private extraction benchmark was run. Linux build/runtime verification belongs to the platform CI and release validation; Windows tests alone do not establish a Linux pass. The private e-Archive and OKC accuracy gate remains pending the user's dataset readiness confirmation.
