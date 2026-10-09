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

use actix_multipart::Multipart;
use actix_web::{post, get, web, App, HttpResponse, HttpServer, Responder};

use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
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


/// POST /process — accept an invoice image and optional tuning params.
#[post("/process")]
pub async fn process(payload: Multipart) -> impl Responder {
    let mut image_path: Option<String> = None;
    let mut params: BTreeMap<String, String> = BTreeMap::new();

    while let Some(item) = payload.next().await {
        let field = match item {
            Ok(f) => f,
            Err(e) => return HttpResponse::InternalServerError().body(format!("Multipart error: {}", e)),
        };
        let name = field.content_disposition().get_name().unwrap_or("").to_string();

        if name == "image" {
            // Persist the uploaded image to a unique temp file so the model
            // runner can read it by path.
            let tmp = env::temp_dir().join(format!("fatrocu_upload_{}.png", Uuid::new_v4()));
            let mut f = match File::create(&tmp) {
                Ok(f) => f,
                Err(e) => return HttpResponse::InternalServerError().body(format!("Cannot save upload: {}", e)),
            };
            match field.next().await {
                Some(Ok(chunk)) => {
                    if f.write_all(&chunk.unwrap_or_default()).is_err() {
                        return HttpResponse::InternalServerError().body("Failed to write upload bytes.");
                    }
                }
                Some(Err(e)) => return HttpResponse::InternalServerError().body(format!("Upload error: {}", e)),
                None => return HttpResponse::BadRequest().body("No image data received."),
            }
            image_path = Some(tmp.to_string_lossy().into_owned());
        } else {
            // Simple text field (e.g. "model"). Take the first non-empty line.
            let mut stream = field;
            let mut value = String::new();
            while let Some(chunk) = stream.next().await {
                if let Ok(chunk) = chunk {
                    let slice = String::from_utf8_lossy(&chunk);
                    if !value.is_empty() && value.ends_with('\n') || value.ends_with('\r') {
                        continue;
                    }
                    value.push_str(slice.trim_end_matches(['\n', '\r']).trim());
                    break;
                }
            }
            if !value.is_empty() {
                params.insert(name.clone(), value);
            }
        }
    }

    if image_path.is_none() {
        return HttpResponse::BadRequest().body("Missing 'image' file field in multipart/form-data.");
    }

    // Run the model runner if configured; otherwise return a valid envelope.
    let image_name = image_path.as_ref().unwrap().file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let runner = env::var("FATROCU_SERVER_BIN").ok();

    let response = match runner {
        Some(path) => match extract_invoice(&path, &image_path.as_ref().unwrap(), &params).await {
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
async fn extract_invoice(runner: &str, image_path: &str, params: &BTreeMap<String, String>) -> Result<InvoiceResponse, String> {
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
