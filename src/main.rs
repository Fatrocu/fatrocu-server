use actix_multipart::Multipart;
use actix_web::post;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};


use futures_util::stream::StreamExt;
use uuid::Uuid;

use serde::Deserialize;
use dotenv::dotenv;


use std::env;
use std::process::Command;

#[derive(Deserialize)]
struct ProcessParams {
    model: String,
    temp: f64,
    n_predict: usize,
    ctx_size: usize,
}

#[post("/process")]
async fn process(mut payload: Multipart) -> impl Responder {
    // Extract fields from multipart/form-data
    let mut image_path = String::new();
    let mut params = ProcessParams {
        model: "ImajeV-2B-Q8_0".into(),
        temp: 0.2,
        n_predict: 8,
        ctx_size: 256,
    };

    while let Some(item) = payload.next().await {
        let mut field = match item {
            Ok(f) => f,
            Err(e) => return HttpResponse::InternalServerError().body(format!("Multipart error: {}", e)),
        };
        let name = {
            let cd = field.content_disposition();
            cd.get_name().unwrap_or("").to_string()
        };

        if name == "image" {
            // Save the uploaded image to a temp file
            let tmp = env::temp_dir().join(format!("upload_{}.png", Uuid::new_v4()));
            let mut f = std::fs::File::create(&tmp).unwrap();
            while let Some(chunk) = field.next().await {
                let data = chunk.unwrap();
                use std::io::Write;
                f.write_all(&data).unwrap();
            }
            image_path = tmp.to_string_lossy().to_string();
        } else {
            // Simple text fields
            let mut bytes = web::BytesMut::new();
            let mut stream = field;
            while let Some(chunk) = stream.next().await {
                bytes.extend_from_slice(&chunk.unwrap());
            }
            let value = std::str::from_utf8(&bytes).unwrap();
            match name.as_str() {
                "model" => params.model = value.to_string(),
                "temp" => params.temp = value.parse().unwrap_or(0.2),
                "n_predict" => params.n_predict = value.parse().unwrap_or(8),
                "ctx_size" => params.ctx_size = value.parse().unwrap_or(256),
                _ => {}
            }
        }
    }

    // Build llama‑cli command
    let output = Command::new("C:/Users/PC/Desktop/fatrocu-cli/llama-cli.exe")
        .args(&[
            "--model",
            &format!("C:/Users/PC/Desktop/fatrocu-cli/models/{}.gguf", params.model),
            "--mmproj",
            &format!("C:/Users/PC/Desktop/fatrocu-cli/models/{}-mmproj-f16.gguf", params.model),
            "--image",
            &image_path,
            "--temp",
            &params.temp.to_string(),
            "--threads",
            "4",
            "--gpu-layers",
            "0",
            "--n-predict",
            &params.n_predict.to_string(),
            "--ctx-size",
            &params.ctx_size.to_string(),
        ])
        .output()
        .expect("failed to run llama-cli");

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return HttpResponse::InternalServerError().body(err.to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    HttpResponse::Ok().content_type("application/json").body(stdout.into_owned())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    let port: u16 = env::var("FATROCU_SERVER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(1842);
    println!("🚀 fatrocu‑server listening on 0.0.0.0:{}", port);
    HttpServer::new(|| App::new().service(process))
        .bind(("0.0.0.0", port))?
        .run()
        .await
}
