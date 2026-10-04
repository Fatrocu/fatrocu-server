# Fatrocu Server

Fatrocu Server provides a lightweight HTTP service for processing invoice images using the local `llama-cli` model. It listens on port 1842 by default and exposes a single endpoint:

- `POST /process` – multipart/form-data with fields `image`, `model`, `temp`, `n_predict`, `ctx_size`.

The server forwards the request to the `llama-cli` executable, runs the specified model and returns the raw JSON output.

## Building

```sh
cd C:/Users/PC/Desktop/fatrocu-server
cargo build --release
```

The binary is placed at `target/release/fatrocu-server.exe`.

## Running

```sh
# Default port 1842
target\release\fatrocu-server.exe

# Custom port
FATROCU_SERVER_PORT=8080 target\release\fatrocu-server.exe
```

The service binds to `0.0.0.0` and logs the listening address.

## API

`POST /process`

The request must be `multipart/form-data` containing:

- **image** – the image file to process.
- **model** (optional) – model identifier, defaults to the built‑in model.
- **temp** (optional) – sampling temperature, default `0.2`.
- **n_predict** (optional) – maximum tokens, default `8`.
- **ctx_size** (optional) – context window size, default `256`.

The server constructs a `llama-cli` command similar to:

```
llama-cli \
  --model <model>.gguf \
  --mmproj <model>-mmproj-f16.gguf \
  --image <temp file> \
  --temp <temp> \
  --threads 4 \
  --gpu-layers 0 \
  --n-predict <n_predict> \
  --ctx-size <ctx_size>
```

The command’s stdout is returned as the HTTP response with `Content-Type: application/json`. Errors are returned as HTTP 500 with the error message.

## Configuration

- `FATROCU_SERVER_PORT` – TCP port to listen on (default `1842`).
- Ensure `llama-cli.exe` is available in the system `PATH` or adjust the source code to point to its location.

## License

MIT License – see the `LICENSE` file.
