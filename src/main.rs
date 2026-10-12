use actix_web::body::MessageBody;
use actix_web::middleware::Logger;
use actix_web::post;
use actix_web::web::Query;
use actix_web::{get, HttpResponse, Responder};
use actix_web::App;

use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const CANONICAL_MODEL: &str = "İmajeV-2B-Q8_0";
const DEFAULT_PORT: u16 = 1842;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessedDocument {
    page: Option<String>,
    fatura_no: Option<String>,
    fatura_tarihi: Option<String>,
    cari_unvan: Option<String>,
    cari_vergi_no: Option<String>,
    ettn_uuid: Option<String>,
    mal_hizmet_toplam_matrah: f64,
    kdv_orani: f64,
    kdv_tutari: f64,
    genel_toplam: f64,
    kalemler: Vec<LineItem>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LineItem {
    fatura_no: Option<String>,
    kdv_matrahi: f64,
    kdv_tutari: f64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    canonical_model: String,
    version: &'static str,
    runner: Option<String>,
}

#[derive(Debug, Serialize)]
struct InvoicesResponse {
    count: usize,
    invoices: Vec<ProcessedDocument>,
}

#[derive(Debug, Deserialize)]
struct TextParams {
    #[serde(default)]
    model: String,
    #[serde(default = "default_temp")]
    temp: f64,
    #[serde(default = "default_n_predict")]
    n_predict: u32,
    #[serde(default = "default_ctx_size")]
    ctx_size: u32,
}

fn default_temp() -> f64 {
    0.2
}

fn default_n_predict() -> u32 {
    8
}

fn default_ctx_size() -> u32 {
    256
}

fn runner_path() -> Option<String> {
    std::env::var("FATROCU_SERVER_BIN").ok().filter(|p| !p.is_empty())
}

fn extract_field(body: &[u8], name: &str) -> Option<String> {
    let header = body.to_ascii_lowercase();
    let marker = format!("name=\"{}\"", name);
    let start = header.find(&marker)?;
    let after_marker = start + marker.len();
    let end_marker = format!(";filename=\"");
    let end = header[after_marker..]
        .find(&end_marker)
        .map(|o| after_marker + o)
        .unwrap_or(header.len());
    Some(String::from_utf8_lossy(&header[after_marker..end]).into_owned())
}

fn split_fields(body: &[u8]) -> Vec<(String, Vec<u8>)> {
    let boundary = extract_field(body, "boundary").unwrap_or_default();
    if boundary.is_empty() {
        return Vec::new();
    }
    let mut b = body.to_vec();
    let prefix = format!("--{}", boundary);
    let suffix = format!("--{}--", boundary);
    let mut results = Vec::new();
    let mut rest = &b;
    while rest.starts_with(prefix.as_bytes()) {
        rest = &rest[prefix.len()..];
        if let Some(rest) = rest.strip_prefix(b"\r\n") {
            let (head, tail) = rest.split_once(b"\r\n--").unwrap_or((&rest[..], &[][..]));
            let name = extract_field(head, "name").unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let bytes = head.replace("Content-Disposition:", "").replace("Content-Type:", "");
            results.push((name, bytes));
            rest = tail;
        } else {
            break;
        }
    }
    results
}

fn extract_invoice(fields: &[(String, Vec<u8>)]) -> ProcessedDocument {
    let find = |n: &str| {
        fields
            .iter()
            .find(|(k, _)| k == n)
            .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
    };
    let find_f64 = |n: &str| {
        fields
            .iter()
            .find(|(k, _)| k == n)
            .and_then(|(_, v)| v.trim().parse::<f64>().ok())
    };
    ProcessedDocument {
        page: None,
        fatura_no: find("faturaNumarasi"),
        fatura_tarihi: find("faturaTarihi"),
        cari_unvan: find("saticiUnvan"),
        cari_vergi_no: find("saticiVknTckn"),
        ettn_uuid: find("ettn_uuid"),
        mal_hizmet_toplam_matrah: find_f64("genel_toplam").unwrap_or(0.0),
        kdv_orani: find_f64("kdvOrani").unwrap_or(0.0),
        kdv_tutari: find_f64("kdvTutari").unwrap_or(0.0),
        genel_toplam: find_f64("genel_toplam").unwrap_or(0.0),
        kalemler: Vec::new(),
    }
}

fn save_upload(image_data: &[u8]) -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let uuid = Uuid::new_v4();
    let path = dir.join(format!("fatrocu_upload_{}_{}.png", stamp, uuid));
    std::fs::write(&path, image_data)?;
    Ok(path)
}

async fn run_llama(
    image_path: &PathBuf,
    model: &str,
    temp: f64,
    n_predict: u32,
    ctx_size: u32,
) -> Result<String, String> {
    let runner = runner_path();
    match &runner {
        Some(path) if std::path::Path::new(path).exists() => {
            let output = Command::new(path)
                .arg("--model")
                .arg(model)
                .arg(image_path)
                .arg("--temp")
                .arg(format!("{temp}"))
                .arg("--threads")
                .arg("4")
                .arg("--gpu-layers")
                .arg("0")
                .arg("--ctx-size")
                .arg(format!("{ctx_size}"))
                .arg("--n-predict")
                .arg(format!("{n_predict}"))
                .output()
                .map_err(|e| format!("runner failed: {e}"))?;
            if !output.status.success() {
                return Err("runner exited with error".to_string());
            }
            String::from_utf8(output.stdout.clone())
                .map_err(|_| "runner produced non-utf8 output".to_string())
        }
        _ => Err("FATROCU_SERVER_BIN not set or runner not found".to_string()),
    }
}

async fn parse_and_extract(
    image_data: &[u8],
    params: &TextParams,
) -> Result<ProcessedDocument, String> {
    let image_path = save_upload(image_data)?;
    let runner = runner_path();
    match &runner {
        Some(path) if std::path::Path::new(path).exists() => {
            let raw = run_llama(
                &image_path,
                &params.model,
                params.temp,
                params.n_predict,
                params.ctx_size,
            )
            .await
            .map_err(|e| format!("extraction failed: {e}"))?;
            serde_json::from_str::<ProcessedDocument>(&raw)
                .map_err(|e| format!("parse failed: {e}"))
        }
        _ => Err("FATROCU_SERVER_BIN not set or runner not found".to_string()),
    }
}

#[derive(Debug)]
struct Service {
    invoices: Vec<ProcessedDocument>,
}

impl Default for Service {
    fn default() -> Self {
        Self {
            invoices: Vec::new(),
        }
    }
}

async fn health() -> impl Responder {
    let runner = runner_path();
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
        model: CANONICAL_MODEL.to_string(),
        canonical_model: CANONICAL_MODEL.to_string(),
        version: env!("CARGO_PKG_VERSION"),
        runner: runner.as_deref().map(|s| s.to_string()),
    })
}

#[post("/process")]
async fn process(
    body: impl MessageBody,
    params: actix_web::web::Query<TextParams>,
) -> Result<impl Responder, String> {
    let image_data = body.into_data().await
        .map_err(|e| format!("read error: {e}"))?;
    if image_data.is_empty() {
        return Err("no image data".to_string());
    }
    let doc = parse_and_extract(&image_data, &params)
        .await
        .map_err(|e| format!("extraction error: {e}"))?;
    crate::AppState::instance(uuid::Uuid::new_v4().as_u128())
        .set_invoice(doc.clone());
    HttpResponse::Ok().json(doc)
}

async fn list() -> Result<impl Responder, String> {
    let invoices = crate::AppState::instance(0).get_invoices();
    let resp = InvoicesResponse {
        count: invoices.len(),
        invoices: invoices.clone(),
    };
    Ok(HttpResponse::Ok().json(resp))
}

async fn export() -> Result<impl Responder, String> {
    let invoices = crate::AppState::instance(0).get_invoices();
    let file = save_xlsx(&invoices)?;
    Ok(HttpResponse::Ok()
        .content_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        .attachment("Fatrocu_Raporu.xlsx", file))
}

fn save_xlsx(invoices: &[ProcessedDocument]) -> std::io::Result<PathBuf> {
    use rust_xlsxwriter::{Sheet, Workbook};
    let path = std::path::PathBuf::from("Fatrocu_Raporu.xlsx");
    let mut workbook = Workbook::new();
    let mut sheet = workbook.add_worksheet()?;
    let mut row = 0i32;
    sheet.write(0, 0, "Page")?;
    sheet.write(0, 1, "Fatura No")?;
    sheet.write(0, 2, "Fatura Tarihi")?;
    sheet.write(0, 3, "Satıcı Ünvan")?;
    sheet.write(0, 4, "Satıcı VKN/TCKN")?;
    sheet.write(0, 5, "Matrah")?;
    sheet.write(0, 6, "KDV Oranı")?;
    sheet.write(0, 7, "KDV Tutarı")?;
    sheet.write(0, 8, "Genel Toplam")?;
    sheet.write(0, 9, "ETTN UUID")?;
    row += 1;
    for inv in invoices {
        let p = inv.page.clone().unwrap_or_default();
        let n = inv.fatura_no.clone().unwrap_or_default();
        let t = inv.fatura_tarihi.clone().unwrap_or_default();
        let u = inv.cari_unvan.clone().unwrap_or_default();
        let v = inv.cari_vergi_no.clone().unwrap_or_default();
        let m = inv.mal_hizmet_toplam_matrah;
        let k = inv.kdv_orani;
        let d = inv.kdv_tutari;
        let g = inv.genel_toplam;
        let e = inv.ettn_uuid.clone().unwrap_or_default();
        sheet.write(row, 0, &p)?;
        sheet.write(row, 1, &n)?;
        sheet.write(row, 2, &t)?;
        sheet.write(row, 3, &u)?;
        sheet.write(row, 4, &v)?;
        sheet.write(row, 5, &m)?;
        sheet.write(row, 6, &k)?;
        sheet.write(row, 7, &d)?;
        sheet.write(row, 8, &g)?;
        sheet.write(row, 9, &e)?;
        row += 1;
    }
    let mut opts = Sheet::default();
    opts.set_column_width(0, 0, 60)?;
    sheet.finish(&mut opts)?;
    sheet.finish(&mut workbook)?;
    workbook.write_to_file(&path)?;
    Ok(path)
}
    let mut row = 0i32;
    sheet.write(0, 0, "Page")?;
    sheet.write(0, 1, "Fatura No")?;
    sheet.write(0, 2, "Fatura Tarihi")?;
    sheet.write(0, 3, "Satıcı Ünvan")?;
    sheet.write(0, 4, "Satıcı VKN/TCKN")?;
    sheet.write(0, 5, "Matrah")?;
    sheet.write(0, 6, "KDV Oranı")?;
    sheet.write(0, 7, "KDV Tutarı")?;
    sheet.write(0, 8, "Genel Toplam")?;
    sheet.write(0, 9, "ETTN UUID")?;
    row += 1;
    for inv in invoices {
        let p = inv.page.clone().unwrap_or_default();
        let n = inv.fatura_no.clone().unwrap_or_default();
        let t = inv.fatura_tarihi.clone().unwrap_or_default();
        let u = inv.cari_unvan.clone().unwrap_or_default();
        let v = inv.cari_vergi_no.clone().unwrap_or_default();
        let m = inv.mal_hizmet_toplam_matrah;
        let k = inv.kdv_orani;
        let d = inv.kdv_tutari;
        let g = inv.genel_toplam;
        let e = inv.ettn_uuid.clone().unwrap_or_default();
        sheet.write(row, 0, &p)?;
        sheet.write(row, 1, &n)?;
        sheet.write(row, 2, &t)?;
        sheet.write(row, 3, &u)?;
        sheet.write(row, 4, &v)?;
        sheet.write(row, 5, &m)?;
        sheet.write(row, 6, &k)?;
        sheet.write(row, 7, &d)?;
        sheet.write(row, 8, &g)?;
        sheet.write(row, 9, &e)?;
        row += 1;
    }
    let mut opts = Sheet::default();
    opts.set_column_width(0, 0, 60)?;
    sheet.finish(&mut opts)?;
    sheet.finish(&mut workbook)?;
    workbook.finish()?;
    Ok(std::path::PathBuf::from("Fatrocu_Raporu.xlsx"))
}

fn main() {
    let _ = dotenv().ok();
    let addr = format!("0.0.0.0:{}", std::env::var("FATROCU_SERVER_PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string()));
    println!("Fatrocu Server listening on {addr}");
    println!("Canonical model: {CANONICAL_MODEL}");
    let app = App::new().service(health).service(process).service(list).service(export);
    actix_web::HttpServer::new(move || {
        let app = app.clone();
        Logger::default().wrap(app)
    })
    .listen(addr)
    .unwrap_or_else(|e| panic!("server bootstrap failed: {e}"));
}
