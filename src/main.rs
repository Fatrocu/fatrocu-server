// Fatrocu Server — pure HTTP REST API for invoice extraction.
//
// Decoupled design: this binary exposes HTTP endpoints only. It never shells
// out to a local model binary and carries no hardcoded user paths, so it can
// be started on any machine and pointed at a model by its name.
//
// Model running is delegated to an optional external runner binary whose
// absolute path is supplied at build time via the FATROCU_SERVER_BIN
// environment variable. If that variable is unset, the server still boots and
// reports an "unconfigured" runner status; callers simply cannot run yet.

use actix_web::body::MessageBody;
use actix_web::dev::{Payload, PayloadReader};
use actix_web::http::header::{ContentType, ImageFormat};
use actix_web::web::{self, Bytes};
use actix_web::{post, get, HttpMessage, HttpResponse};
use actix_web::{Error, HttpRequest};
use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const CANONICAL_MODEL: &str = "İmajeV-2B-Q8_0";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Runner binary path, injected at build time. Optional: unset => unconfigured.
fn runner_path() -> Option<PathBuf> {
    env::var("FATROCU_SERVER_BIN")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
}

/// Reads the whole request body into a Vec<u8>.
async fn read_body(req: &HttpRequest) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = req.body().read(&mut buf).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }
    Ok(body)
}

/// Splits a multipart/form-data body into (name, value) pairs.
fn parse_multipart(body: &[u8], boundary: &str) -> HashMap<String, String> {
    let mut parts: HashMap<String, String> = HashMap::new();
    let marker = format!("\r\n--{boundary}--");
    let chunks = body.split(format!("--{boundary}"));

    for chunk in chunks {
        let chunk = chunk.trim();
        let chunk = chunk.strip_suffix(marker.as_bytes()).unwrap_or(chunk);
        let chunk = chunk.strip_suffix(b"\r\n").unwrap_or(chunk);

        let header_end = match chunk.iter().position(|&b| b == b'\r') {
            Some(i) => i,
            None => continue,
        };
        let header = &chunk[..header_end];
        let header = String::from_utf8_lossy(header);

        if let Some(pos) = header.find("name=\"") {
            let end = header[pos + "name=\"".len()..]
                .find('"')
                .map(|i| pos + "name=\"".len() + i)
                .unwrap_or(0);
            let name = &header[header.find("name=").unwrap_or(0) + "name=".len()..end];
            let value = &chunk[header_end + 2..];
            if let Some(value_end) = value.iter().position(|&b| b == b'\r') {
                let value = &value[..value_end];
                parts.insert(
                    name.to_string(),
                    String::from_utf8_lossy(value).to_string(),
                );
            }
        }
    }
    parts
}

/// Resolves the configured runner path (or a friendly message if unset).
fn resolve_runner() -> Result<(PathBuf, String), Error> {
    match runner_path() {
        Some(p) => {
            let _ = std::fs::metadata(&p);
            Ok((p, String::new()))
        }
        None => Err(Error::InternalServerError(
            "Fatrocu runner is not configured: set FATROCU_SERVER_BIN to the path of the Fatrocu model runner binary",
        )),
    }
}

/// Returns the model name used by the runner, defaulting to the canonical name.
fn model_name() -> String {
    let mut name = env::var("FATROCU_MODEL_NAME")
        .unwrap_or_default()
        .trim()
        .to_string();
    if name.is_empty() {
        name = CANONICAL_MODEL.to_string();
    }
    name
}

#[get("/health")]
async fn health() -> HttpResponse {
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "model": model_name(),
        "canonical_model": CANONICAL_MODEL,
        "version": SERVER_VERSION,
        "runner": match runner_path() {
            Some(p) => format!("{} (configured)", p.display()),
            None => "unconfigured".to_string(),
        },
    }))
}

/// POST /process — accept a single image file plus optional text fields,
/// pass them to the runner, and return the JSON extraction result.
#[post("/process")]
async fn process(
    req: HttpRequest,
    payload: impl MessageBody,
) -> Result<HttpResponse, Error> {
    let content_type = req
        .content_type()
        .unwrap_or("application/octet-stream")
        .to_string();

    // Multipart body (image file + text params) or a single JSON payload.
    let parsed = match read_body(&req).await {
        Ok(bytes) => parse_multipart(&bytes, &content_type).ok(),
        Err(_) => None,
    };

    // Fall back to a raw JSON body if multipart parsing failed.
    let fields: HashMap<String, String> = if parsed.is_none() {
        serde_json::from_slice::<serde_json::Value>(&read_body(&req).await.unwrap_or_default())
            .ok()
            .and_then(|v| {
                v.as_object()
                    .map(|m| m.iter().map(|(k, val)| (k.clone(), val.as_str().unwrap_or("").to_string())).collect())
            })
    } else {
        parsed
    };

    let fields = match fields {
        Some(f) => f,
        None => return Err(Error::BadRequest("no form fields found in request body")),
    };

    let image_name = fields
        .get("image")
        .or_else(|| fields.get("file"))
        .or_else(|| fields.get("image_file"))
        .cloned()
        .ok_or_else(|| Error::BadRequest("missing image file field (expected 'image' or 'file')"))?;

    let model = fields.get("model").cloned().unwrap_or_else(|| model_name());
    let temp = fields.get("temp").and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(0.2);
    let n_predict = fields
        .get("n_predict")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(8);
    let ctx_size = fields
        .get("ctx_size")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(256);

    let tmp_path = format!(
        "fatrocu_upload_{}_{}.png",
        model,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    );
    let tmp_path = PathBuf::from(&tmp_path);
    let parent = tmp_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir());
    let _ = std::fs::create_dir_all(parent);

    // Save the uploaded image bytes to a temp file so the runner can read them.
    let image_data = match &image_name {
        Some(name) => {
            let part = parse_multipart(&read_body(&req).await.unwrap_or_default(), &content_type);
            if let Some(bytes) = part.get(name).map(String::as_bytes).flatten() {
                File::create(&tmp_path)
                    .and_then(|mut f| f.write_all(bytes))
                    .and_then(|_| f.flush())
                    .map_err(|e| Error::InternalServerError(format!("failed to save image: {e}")))?;
                Some(&tmp_path)
            } else {
                None
            }
        }
        None => None,
    };

    // Build the runner argument list: model + image + tunables.
    let mut args: Vec<String> = vec![model.clone()];
    if let Some(path) = image_data {
        args.push(path.display().to_string());
    }
    args.push("--temp".into());
    args.push(temp.to_string());
    args.push("--n_predict".into());
    args.push(n_predict.to_string());
    args.push("--ctx_size".into());
    args.push(ctx_size.to_string());

    // Run the external model runner. If it fails or is missing, return a clear
    // error instead of crashing the server.
    let (_, runner_note) = resolve_runner()?;
    let (stdout, stderr) = match (&image_data, &args) {
        (Some(img), args) => match env::var("FATROCU_SERVER_BIN") {
            Ok(bin) => {
                let mut cmd = std::process::Command::new(&bin);
                cmd.arg(args);
                if let Some(img) = img {
                    cmd.arg(img);
                }
                let output = cmd.output().map_err(|e| {
                    Error::InternalServerError(format!("failed to start runner: {e}"))
                })?;
                (
                    String::from_utf8_lossy(&output.stdout).to_string(),
                    String::from_utf8_lossy(&output.stderr).to_string(),
                )
            }
            Err(_) => (
                String::new(),
                format!(
                    "runner not configured (set FATROCU_SERVER_BIN); model: {}",
                    model
                ),
            ),
        },
        (None, _) => (
            String::new(),
            "no image provided to runner".to_string(),
        ),
    };

    Ok(HttpResponse::Ok().json(json!({
        "model": model,
        "runner_status": runner_note,
        "ok": image_data.is_some(),
        "raw_stdout": stdout,
        "raw_stderr": stderr,
    })))
}

/// GET /invoices — returns the list of previously extracted invoices (stored
/// under a sidecar path so the server stays a pure HTTP service).
#[get("/invoices")]
async fn invoices() -> HttpResponse {
    let home = env::temp_dir();
    let path = home.join("fatrocu_invoices.json");
    match std::fs::read_to_string(&path) {
        Ok(text) => HttpResponse::Ok().json(serde_json::from_str::<Vec<serde_json::Value>>(&text).unwrap_or_default()),
        Err(_) => HttpResponse::Ok().json(Vec::<serde_json::Value>::new()),
    }
}

/// GET /export — writes a real .xlsx report using rust_xlsxwriter.
#[get("/export")]
async fn export_xlsx() -> HttpResponse {
    let home = env::temp_dir();
    let path = home.join("Fatrocu_Raporu.xlsx");
    {
        let file = match File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                return HttpResponse::InternalServerError().body(format!("failed to create report: {e}"))
            }
        };
        let mut writer = rust_xlsxwriter::Writer::new(file).unwrap();
        let mut ws = writer.add_worksheet("Invoices").unwrap();
        let _ = ws.write_string(0, "Fatrocu Report").unwrap();
        let _ = ws.write_string(1, "model").unwrap();
        let _ = ws.write_string(2, CANONICAL_MODEL).unwrap();
        let _ = writer.save().unwrap();
    }
    HttpResponse::Ok().attachment("Fatrocu_Raporu.xlsx").body(path.display().to_string())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let _ = actix_web::dotenv::dotenv();

    let port = env::var("FATROCU_SERVER_PORT")
        .ok()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(1842);

    println!("Fatrocu server starting on 0.0.0.0:{port}");

    let app = web::App::new()
        .wrap(Logger::default())
        .route("/health", web::get().to(health))
        .route("/process", web::post().to(process))
        .route("/invoices", web::get().to(invoices))
        .route("/export", web::get().to(export_xlsx));

    actix_web::HttpServer::new(|| {
        let app = app.clone();
        app
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
