//! fatrocu-server — a pure-HTTP API for Fatrocu invoice processing.
//!
//! This service is deliberately **decoupled**: it exposes an HTTP contract and
//! never shells out to a local model binary or hard-codes host paths. This makes
//! it trivial to boot on any platform (just `cargo run`) and easy to integrate
//! from the desktop app, the CLI, or third-party clients.
//!
//! Endpoints:
//!   GET  /health              liveness + model metadata
//!   POST /process             upload an invoice image (multipart/form-data)
//!   GET  /invoices            list invoices saved on disk (requires export)
//!
//! The actual vision/OCR work is delegated to a model runner. When the runner is
//! available it is invoked via the `FATROCU_SERVER_BIN` environment variable
//! (absolute or relative path to an executable). When it is not configured the
//! endpoint still returns a well-formed response envelope so the API contract is
//! always valid.

use actix_web::{get, web, App, HttpResponse, HttpServer, HttpRequest, Responder};

use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use dotenv::dotenv;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Canonical model used by every Fatrocu component.
pub const CANONICAL_MODEL: &str = "İmajeV-2B-Q8_0";

/// Fields accepted as text inputs on `/process`.
#[derive(Deserialize)]
pub struct ProcessParams {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_temp")]
    pub temp: f64,
    #[serde(default = "default_n_predict")]
    pub n_predict: usize,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: usize,
}

fn default_model() -> String {
    CANONICAL_MODEL.to_string()
}

fn default_temp() -> f64 {
    0.2
}

fn default_n_predict() -> usize {
    8
}

fn default_ctx_size() -> usize {
    256
}

/// Structured response envelope returned by `/process`. The field names match
/// the desktop app's `ProcessedInvoice` schema so responses are directly usable.
#[derive(Serialize)]
pub struct InvoiceResponse {
    /// Server-side unique id for this response.
    id: String,
    /// Uploaded image file name (base name only).
    image: String,
    /// Requested model identifier.
    model: String,
    /// Whether a real model runner produced the fields below.
    model_loaded: bool,
    /// Processing device (e.g. "CPU", "GPU (N layers)").
    device: String,
    /// Human-readable note, e.g. why fields are empty.
    message: String,
    /// Header fields of the invoice.
    fatura_no: Option<String>,
    fatura_tarihi: Option<String>,
    fatura_turu: Option<String>,
    cariler: Vec<LineItem>,
    genel_toplam: Option<f64>,
    kdv_toplam: Option<f64>,
    matrah_toplam: Option<f64>,
}

#[derive(Serialize)]
pub struct LineItem {
    aciklama: Option<String>,
    kdv_orani: Option<f64>,
    kdv_tutari: Option<f64>,
    matrah: Option<f64>,
    tutar: Option<f64>,
}

/// Health / liveness endpoint.
#[get("/health")]
pub async fn health() -> HttpResponse {
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "model": CANONICAL_MODEL,
        "version": env!("CARGO_PKG_VERSION"),
        "runner": env::var("FATROCU_SERVER_BIN").map(|p| PathBuf::from(&p).exists()).unwrap_or(false),
    }))
}


/// Read the entire request body into memory.
fn read_body(req: &HttpRequest) -> Result<Vec<u8>, HttpResponse> {
    let mut body = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match req.body().read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => body.extend_from_slice(&buf[..n]),
            Err(e) => return Err(HttpResponse::BadRequest().body(format!("Body read error: {}", e))),
        }
    }
    Ok(body)
}

/// A single multipart part: field name, raw content-type, and raw bytes.
struct Part {
    name: String,
    content_type: String,
    data: Vec<u8>,
}

/// Parse a `multipart/form-data` body. Returns an error string if the boundary
/// is missing so callers can answer 400.
fn parse_multipart(body: &[u8], content_type: &str) -> Result<Vec<Part>, String> {
    // content-type: multipart/form-data; boundary=----xxxx
    let boundary = content_type
        .to_ascii_lowercase()
        .match_indices("boundary=").nth(1)
        .map(|i| &content_type[i + "boundary=".len()..])
        .ok_or("missing multipart boundary in Content-Type header")?;

    let mut parts = Vec::new();
    let mut rest = body;
    let marker = format!("\r\n--{}", boundary);
    let delimiter = format!("\r\n--{}", boundary);

    loop {
        // Locate the start of the next part.
        let start = match rest.windows(marker.len()).position(|w| w == &marker) {
            Some(i) => i,
            None => break,
        };
        let after_marker = &rest[start + marker.len()..];

        // A part ends at the next "--boundary" (CRLF = more parts, "--" = final).
        let split = match after_marker.find(&delimiter) {
            Some(i) => i,
            None => break,
        };
        let part_bytes = &after_marker[..split];
        rest = &after_marker[split + delimiter.len()..];

        // Headers run until the first blank line.
        let header_end = match part_bytes.find("\r\n\r\n") {
            Some(i) => i,
            None => break,
        };
        let headers = &part_bytes[..header_end];
        let data = part_bytes[header_end + "\r\n\r\n".len()..].to_vec();

        // Extract name="..." from Content-Disposition and the Content-Type.
        let mut name = None;
        let mut content_type = "application/octet-stream".to_string();
        for line in headers.split("\r\n") {
            if let Some((k, v)) = line.split_once(':') {
                let key = k.trim().to_ascii_lowercase();
                let val = v.trim();
                if key == "content-disposition" {
                    if let Some(after) = val.find("name=") {
                        let rest = &val(after + "name=".len());
                        if let Some(open) = rest.find('"') {
                            if let Some(close) = rest[open + 1..].find('"') {
                                let inner = &rest[open + 1..open + 1 + close];
                                name = Some(inner.trim_matches('"').to_string());
                            }
                        }
                    }
                } else if key == "content-type" {
                    content_type = val.to_string();
                }
            }
        }

        parts.push(Part { name: name.unwrap_or_default(), content_type, data });
    }

    Ok(parts)
}

/// POST /process — accept an invoice image (multipart/form-data) and optional
/// tuning params (model/temp/n_predict/ctx_size).
#[post("/process")]
pub async fn process(req: HttpRequest) -> impl Responder {
    let content_type = req
        .content_type()
        .as_deref()
        .unwrap_or("")
        .to_string();

    let parts = match read_body(req) {
        Ok(b) => match parse_multipart(&b, &content_type) {
            Ok(p) => p,
            Err(e) => return HttpResponse::BadRequest().body(e),
        },
        Err(e) => return Err(e),
    };

    let mut image_path: Option<String> = None;
    let mut params: BTreeMap<String, String> = BTreeMap::new();

    for part in &parts {
        if part.name.is_empty() {
            continue;
        }
        if part.content_type.to_ascii_lowercase().contains("image") {
            // Persist the uploaded image to a unique temp file for the runner.
            let tmp = env::temp_dir().join(format!("fatrocu_upload_{}.png", Uuid::new_v4()));
            if let Err(e) = File::create(&tmp).and_then(|mut f| f.write_all(&part.data)).and_then(|_| f.flush()) {
                return HttpResponse::InternalServerError().body(format!("Cannot save upload: {}", e));
            }
            image_path = Some(tmp.to_string_lossy().into_owned());
        } else {
            // Text field (e.g. "model"): take the first non-empty line.
            let text = String::from_utf8_lossy(&part.data);
            if let Some(line) = text.lines().find(|l| !l.trim().is_empty()) {
                params.insert(part.name.clone(), line.trim().to_string());
            }
        }
    }

    if image_path.is_none() {
        return HttpResponse::BadRequest().body("Missing 'image' file field in multipart/form-data.");
    }

    // Run the model runner if configured; otherwise return a valid envelope.
    let image_name = Path::new(image_path.as_ref().unwrap()).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let runner = env::var("FATROCU_SERVER_BIN").ok();

    let response = match runner {
        Some(path) => match extract_invoice(&path, &image_path.as_ref().unwrap()).await {
            Ok(r) => r,
            Err(e) => InvoiceResponse {
                id: Uuid::new_v4().to_string(),
                image: image_name,
                model: CANONICAL_MODEL.to_string(),
                model_loaded: false,
                device: "Unknown".to_string(),
                message: format!("Model runner invocation failed: {}", e),
                fatura_no: None,
                fatura_tarihi: None,
                fatura_turu: None,
                cariler: Vec::new(),
                genel_toplam: None,
                kdv_toplam: None,
                matrah_toplam: None,
            },
        },
        None => InvoiceResponse {
            id: Uuid::new_v4().to_string(),
            image: image_name,
            model: params.get("model").cloned().unwrap_or_else(|| CANONICAL_MODEL.to_string()),
            model_loaded: false,
            device: "CPU".to_string(),
            message: "No model runner configured. Set FATROCU_SERVER_BIN to a running Fatrocu CLI/runner to extract invoice fields."
                .to_string(),
            fatura_no: None,
            fatura_tarihi: None,
            fatura_turu: None,
            cariler: Vec::new(),
            genel_toplam: None,
            kdv_toplam: None,
            matrah_toplam: None,
        },
    };

    HttpResponse::Ok().content_type("application/json").body(serde_json::to_string(&response).unwrap())
}

/// Invoke the model runner binary with the image and return parsed invoice JSON.
async fn extract_invoice(runner: &str, image_path: &str) -> Result<InvoiceResponse, String> {
    // The runner is expected to accept: <model> <image> and print JSON on stdout.
    // This is a platform-neutral invocation; when unavailable we fail gracefully.
    if !Path::new(runner).exists() {
        return Err("runner binary not found at FATROCU_SERVER_BIN".to_string());
    }

    let output = std::process::Command::new(runner)
        .arg(CANONICAL_MODEL)
        .arg(image_path)
        .output()
        .map_err(|e| format!("failed to run runner: {}", e))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map(|v| InvoiceResponse {
            id: Uuid::new_v4().to_string(),
            image: image_path.clone(),
            model: CANONICAL_MODEL.to_string(),
            model_loaded: true,
            device: "GPU (0)".to_string(),
            message: "Processed by the configured model runner.".to_string(),
            fatura_no: v.get("fatura_no").map(|s| s.as_str().unwrap().to_string()),
            fatura_tarihi: v.get("fatura_tarihi").map(|s| s.as_str().unwrap().to_string()),
            fatura_turu: v.get("fatura_turu").map(|s| s.as_str().unwrap().to_string()),
            cariler: v
                .get("cariler")
                .and_then(|a| a.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|k| LineItem {
                            aciklama: k.get("aciklama").map(|s| s.as_str().unwrap().to_string()),
                            kdv_orani: k.get("kdv_orani").and_then(|n| n.as_f64()),
                            kdv_tutari: k.get("kdv_tutari").and_then(|n| n.as_f64()),
                            matrah: k.get("matrah").and_then(|n| n.as_f64()),
                            tutar: k.get("tutar").and_then(|n| n.as_f64()),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            genel_toplam: v.get("genel_toplam").and_then(|n| n.as_f64()),
            kdv_toplam: v.get("kdv_toplam").and_then(|n| n.as_f64()),
            matrah_toplam: v.get("matrah_toplam").and_then(|n| n.as_f64()),
        }),
    Err(format!("runner output is not valid invoice JSON: {}", stdout.trim()))
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    let port: u16 = env::var("FATROCU_SERVER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(1842);
    let runner = env::var("FATROCU_SERVER_BIN").unwrap_or_default();
    println!("🚀 fatrocu‑server listening on 0.0.0.0:{}", port);
    if !runner.is_empty() {
        println!("    model runner: {}", runner);
    } else {
        println!("    model runner: not set (FATROCU_SERVER_BIN) — /process returns an envelope, not extracted fields");
    }
    HttpServer::new(|| {
        App::new()
            .route("/health", web::get().to(health))
            .route("/process", web::post().to(process))
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
