use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use fatrocu_core::{DEFAULT_MODEL, update::verify_digest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    net::IpAddr,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

const RUNTIME_TAG: &str = "b12069";
const MODEL_REVISION: &str = "e87f176479d0855a907a41277aca2f8ee7a09523";
#[derive(Parser)]
#[command(
    version,
    about = "Install and manage a local llama.cpp multimodal model server"
)]
struct Cli {
    #[arg(long, env = "FATROCU_SERVER_DATA", global = true)]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Commands>,
}
#[derive(Subcommand)]
enum Commands {
    /// Standard installs the verified CPU runtime and default multimodal model.
    Setup {
        #[arg(long)]
        advanced: bool,
        #[arg(long,default_value=DEFAULT_MODEL)]
        model: String,
        #[arg(long)]
        revision: Option<String>,
        #[arg(long)]
        model_file: Option<String>,
        #[arg(long)]
        projector_file: Option<String>,
        #[arg(long,default_value=RUNTIME_TAG)]
        runtime_tag: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        #[arg(long, default_value_t = 1842)]
        port: u16,
        #[arg(long)]
        allow_network_exposure: bool,
        #[arg(long)]
        api_key_env: Option<String>,
        #[arg(long)]
        desktop_script: bool,
        #[arg(long)]
        no_start: bool,
    },
    Start,
    Status,
    /// Display or save an advanced configuration without downloading files.
    Configure {
        file: PathBuf,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    runtime: PathBuf,
    model_path: PathBuf,
    projector_path: PathBuf,
    model_id: String,
    model_revision: String,
    runtime_tag: String,
    host: IpAddr,
    port: u16,
    network_exposure_authorized: bool,
    api_key_env: Option<String>,
    context_size: u32,
    gpu_layers: u32,
}
impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.host.is_loopback() || self.network_exposure_authorized,
            "Host mode exposes document inference to other computers; explicit network exposure authorization is required"
        );
        ensure!(
            self.host.is_loopback() || self.api_key_env.as_ref().is_some_and(|s| !s.is_empty()),
            "Host mode requires an API key environment variable"
        );
        ensure!(
            self.port > 0 && self.context_size >= 8192,
            "Use a nonzero port and at least 8192 context tokens"
        );
        Ok(())
    }
    fn arguments(&self) -> Vec<String> {
        vec![
            "-m".into(),
            self.model_path.to_string_lossy().into(),
            "--mmproj".into(),
            self.projector_path.to_string_lossy().into(),
            "--alias".into(),
            self.model_id.clone(),
            "--host".into(),
            self.host.to_string(),
            "--port".into(),
            self.port.to_string(),
            "-c".into(),
            self.context_size.to_string(),
            "-ngl".into(),
            self.gpu_layers.to_string(),
            "--no-webui".into(),
            "--parallel".into(),
            "1".into(),
            "--image-min-tokens".into(),
            "1024".into(),
        ]
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("Fatrocu-server/3.1")
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(3600))
        .build()?)
}
fn run() -> Result<()> {
    let cli = Cli::parse();
    let root = cli.data_dir.unwrap_or(
        directories::ProjectDirs::from("org", "Fatrocu", "Server")
            .context("Cannot locate server data")?
            .data_local_dir()
            .to_owned(),
    );
    std::fs::create_dir_all(&root)?;
    let command = if let Some(command) = cli.command {
        command
    } else {
        println!(
            "Fatrocu model setup\n1. Standard: same-computer CPU runtime and default multimodal model\n2. Advanced: custom models or network hosting\nChoose [1]:"
        );
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let advanced = input.trim() == "2";
        let mut selected_model = DEFAULT_MODEL.to_owned();
        let mut selected_host = "127.0.0.1".parse::<IpAddr>()?;
        let mut api_key_env = None;
        let mut allow_network_exposure = false;
        if advanced {
            selected_model = prompt("Hugging Face owner/repository:quantization", DEFAULT_MODEL)?;
            selected_host = prompt(
                "Bind address (127.0.0.1 for this computer, 0.0.0.0 to host other computers)",
                "127.0.0.1",
            )?
            .parse()?;
            if !selected_host.is_loopback() {
                println!(
                    "Host mode makes this model API reachable by other computers, which can send document contents. Configure firewall and TLS access before exposing it."
                );
                ensure!(
                    prompt("Type HOST to authorize network exposure", "")? == "HOST",
                    "Network exposure was not authorized"
                );
                allow_network_exposure = true;
                api_key_env = Some(prompt(
                    "Environment variable containing the API key",
                    "FATROCU_MODEL_API_KEY",
                )?);
            }
        }
        Commands::Setup {
            advanced,
            model: selected_model,
            revision: None,
            model_file: None,
            projector_file: None,
            runtime_tag: RUNTIME_TAG.into(),
            host: selected_host,
            port: 1842,
            allow_network_exposure,
            api_key_env,
            desktop_script: false,
            no_start: false,
        }
    };
    match command {
        Commands::Setup {
            advanced,
            model,
            revision,
            model_file,
            projector_file,
            runtime_tag,
            host,
            port,
            allow_network_exposure,
            api_key_env,
            desktop_script,
            no_start,
        } => {
            ensure!(
                advanced
                    || (model == DEFAULT_MODEL
                        && host.is_loopback()
                        && api_key_env.is_none()
                        && model_file.is_none()
                        && projector_file.is_none()),
                "Custom models and hosting require --advanced"
            );
            let mut config = Config {
                runtime: PathBuf::new(),
                model_path: PathBuf::new(),
                projector_path: PathBuf::new(),
                model_id: model.clone(),
                model_revision: String::new(),
                runtime_tag: runtime_tag.clone(),
                host,
                port,
                network_exposure_authorized: allow_network_exposure,
                api_key_env,
                context_size: 32768,
                gpu_layers: 0,
            };
            config.validate()?;
            if !host.is_loopback() {
                println!(
                    "Host mode binds to {host}:{port}. Other computers can reach this model API and send document contents. Protect the port with a firewall and authenticated TLS proxy."
                );
            }
            config.runtime = install_runtime(&root, &runtime_tag)?;
            let (repo, quant) = model
                .rsplit_once(':')
                .context("Model must be owner/repository:quantization")?;
            ensure!(
                repo.split('/').count() == 2
                    && repo
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"/_-.".contains(&b)),
                "Invalid Hugging Face repository"
            );
            let requested = revision.unwrap_or_else(|| {
                if model == DEFAULT_MODEL {
                    MODEL_REVISION.into()
                } else {
                    "main".into()
                }
            });
            let metadata: serde_json::Value = client()?
                .get(format!(
                    "https://huggingface.co/api/models/{repo}/revision/{requested}?blobs=true"
                ))
                .send()?
                .error_for_status()?
                .json()?;
            let sha = metadata["sha"].as_str().context("Missing model revision")?;
            config.model_revision = sha.into();
            let siblings = metadata["siblings"]
                .as_array()
                .context("Missing model files")?;
            let choose = |explicit: Option<String>, projector: bool| -> Result<String> {
                if let Some(file) = explicit {
                    return Ok(file);
                }
                let matches: Vec<_> = siblings
                    .iter()
                    .filter_map(|v| v["rfilename"].as_str())
                    .filter(|name| {
                        if projector {
                            *name == "mmproj-F16.gguf"
                        } else {
                            name.ends_with(&format!("-{quant}.gguf"))
                        }
                    })
                    .collect();
                ensure!(
                    matches.len() == 1,
                    "Specify an unambiguous {} filename",
                    if projector {
                        "--projector-file"
                    } else {
                        "--model-file"
                    }
                );
                Ok(matches[0].into())
            };
            let model_file = choose(model_file, false)?;
            let projector_file = choose(projector_file, true)?;
            let model_dir = root.join("models").join(sha);
            std::fs::create_dir_all(&model_dir)?;
            for (name, is_projector) in [(model_file, false), (projector_file, true)] {
                ensure!(
                    Path::new(&name).components().count() == 1 && !Path::new(&name).is_absolute(),
                    "Use a flat model filename"
                );
                let info = siblings
                    .iter()
                    .find(|v| v["rfilename"] == name)
                    .context("Selected model file is absent")?;
                let hash = info
                    .pointer("/lfs/sha256")
                    .and_then(|v| v.as_str())
                    .context("Model file lacks a published SHA-256 digest")?;
                let path = model_dir.join(&name);
                download_verified(
                    &format!("https://huggingface.co/{repo}/resolve/{sha}/{name}"),
                    hash,
                    &path,
                )?;
                if is_projector {
                    config.projector_path = path;
                } else {
                    config.model_path = path;
                }
            }
            save(&root, &config)?;
            if desktop_script {
                launch_script(&root)?;
            }
            println!(
                "Setup complete. Model revision: {}. Endpoint: http://{}:{}/v1",
                config.model_revision, config.host, config.port
            );
            if !no_start {
                start(&config)?;
            }
        }
        Commands::Start => start(&load(&root)?)?,
        Commands::Status => {
            let config = load(&root)?;
            let endpoint = fatrocu_core::inference::Endpoint {
                url: format!("http://127.0.0.1:{}/v1", config.port),
                model: config.model_id.clone(),
                api_key_env: config.api_key_env.clone(),
                ..Default::default()
            };
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"configuration":config,"reachable":endpoint.health().is_ok()})
                )?
            );
        }
        Commands::Configure { file } => {
            let config: Config = serde_json::from_slice(&std::fs::read(file)?)?;
            config.validate()?;
            save(&root, &config)?;
        }
    }
    Ok(())
}
fn prompt(label: &str, default: &str) -> Result<String> {
    print!("{label} [{default}]: ");
    std::io::stdout().flush()?;
    let mut value = String::new();
    ensure!(
        std::io::stdin().read_line(&mut value)? > 0,
        "Input ended; setup cancelled"
    );
    Ok(if value.trim().is_empty() {
        default.into()
    } else {
        value.trim().into()
    })
}
fn save(root: &Path, c: &Config) -> Result<()> {
    let mut temp = tempfile::NamedTempFile::new_in(root)?;
    serde_json::to_writer_pretty(temp.as_file_mut(), c)?;
    temp.as_file().sync_all()?;
    temp.persist(root.join("server.json"))?;
    Ok(())
}
fn load(root: &Path) -> Result<Config> {
    Ok(serde_json::from_slice(
        &std::fs::read(root.join("server.json")).context("Run fatrocu-server setup first")?,
    )?)
}
fn start(c: &Config) -> Result<()> {
    c.validate()?;
    ensure!(
        c.runtime.is_file() && c.model_path.is_file() && c.projector_path.is_file(),
        "Configured runtime/model/projector is missing"
    );
    let mut command = Command::new(&c.runtime);
    command.args(c.arguments());
    if let Some(name) = &c.api_key_env {
        command.env(
            "LLAMA_API_KEY",
            std::env::var(name)
                .with_context(|| format!("Set API key environment variable {name}"))?,
        );
    }
    ensure!(
        command.status()?.success(),
        "llama.cpp server exited with an error"
    );
    Ok(())
}
fn download_verified(url: &str, hash: &str, destination: &Path) -> Result<()> {
    ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "Invalid published digest"
    );
    if destination.exists() {
        let mut file = std::fs::File::open(destination)?;
        let mut digest = Sha256::new();
        std::io::copy(&mut file, &mut digest)?;
        ensure!(
            hex::encode(digest.finalize()) == hash,
            "Existing download digest mismatch; remove the corrupt file before retrying"
        );
        return Ok(());
    }
    println!("Downloading verified artifact...");
    let response = client()?.get(url).send()?.error_for_status()?;
    let mut temp =
        tempfile::NamedTempFile::new_in(destination.parent().context("Missing parent directory")?)?;
    let mut digest = Sha256::new();
    let mut response = response.take(16 * 1024 * 1024 * 1024 + 1);
    let mut chunk = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = response.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        ensure!(total <= 16 * 1024 * 1024 * 1024, "Download exceeds limit");
        digest.update(&chunk[..n]);
        temp.write_all(&chunk[..n])?;
    }
    ensure!(
        hex::encode(digest.finalize()) == hash,
        "Downloaded artifact digest mismatch"
    );
    temp.as_file().sync_all()?;
    temp.persist_noclobber(destination)?;
    Ok(())
}
fn install_runtime(root: &Path, tag: &str) -> Result<PathBuf> {
    ensure!(
        tag.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.'),
        "Invalid runtime release tag"
    );
    let suffix = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => "bin-win-cpu-x64.zip",
        ("windows", "aarch64") => "bin-win-cpu-arm64.zip",
        ("linux", "x86_64") => "bin-ubuntu-x64.tar.gz",
        ("linux", "aarch64") => "bin-ubuntu-arm64.tar.gz",
        _ => anyhow::bail!("Unsupported runtime platform"),
    };
    let dir = root.join("runtime").join(tag);
    if let Some(binary) = find_runtime(&dir) {
        return Ok(binary);
    }
    let metadata: serde_json::Value = client()?
        .get(format!(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
        ))
        .send()?
        .error_for_status()?
        .json()?;
    let expected = format!("llama-{tag}-{suffix}");
    let asset = metadata["assets"]
        .as_array()
        .context("Runtime release has no assets")?
        .iter()
        .find(|a| a["name"] == expected)
        .context("No matching runtime artifact")?;
    let hash = asset["digest"]
        .as_str()
        .and_then(|s| s.strip_prefix("sha256:"))
        .context("Runtime asset lacks a published digest")?;
    let scratch = tempfile::tempdir_in(root)?;
    let archive = scratch.path().join(&expected);
    download_verified(
        asset["browser_download_url"]
            .as_str()
            .context("Missing runtime download URL")?,
        hash,
        &archive,
    )?;
    let staging = scratch.path().join("runtime");
    std::fs::create_dir(&staging)?;
    if suffix.ends_with(".zip") {
        let bytes = std::fs::read(&archive)?;
        verify_digest(&bytes, &format!("sha256:{hash}"))?;
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
        let mut expanded = 0;
        for i in 0..zip.len() {
            let mut f = zip.by_index(i)?;
            let enclosed = f.enclosed_name().context("Unsafe runtime archive path")?;
            ensure!(
                f.unix_mode().is_none_or(|mode| mode & 0o170000 != 0o120000),
                "Runtime symlink rejected"
            );
            expanded += f.size();
            ensure!(
                expanded <= 2 * 1024 * 1024 * 1024,
                "Runtime archive too large"
            );
            let dest = staging.join(enclosed);
            if f.is_dir() {
                std::fs::create_dir_all(dest)?;
            } else {
                std::fs::create_dir_all(dest.parent().unwrap())?;
                let mut out = std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(dest)?;
                std::io::copy(&mut f, &mut out)?;
            }
        }
    } else {
        let file = std::fs::File::open(archive)?;
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
        let mut expanded = 0;
        for entry in archive.entries()? {
            let mut entry = entry?;
            ensure!(
                entry.header().entry_type().is_file()
                    || entry.header().entry_type().is_dir()
                    || entry.header().entry_type().is_symlink(),
                "Runtime devices and hard links rejected"
            );
            if entry.header().entry_type().is_symlink() {
                let target = entry.link_name()?.context("Missing runtime link target")?;
                ensure!(
                    target
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_))),
                    "Runtime links must be relative and stay inside their archive directory"
                );
            }
            expanded += entry.size();
            ensure!(
                expanded <= 2 * 1024 * 1024 * 1024,
                "Runtime archive too large"
            );
            ensure!(entry.unpack_in(&staging)?, "Unsafe runtime archive path");
        }
    }
    ensure!(
        find_runtime(&staging).is_some(),
        "Runtime archive does not contain llama-server"
    );
    std::fs::create_dir_all(dir.parent().unwrap())?;
    std::fs::rename(staging, &dir)?;
    find_runtime(&dir).context("Installed runtime missing")
}
fn find_runtime(dir: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    for candidate in [dir.join(name), dir.join("bin").join(name)] {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        if entry.file_type().ok()?.is_dir() {
            let candidate = entry.path().join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}
fn launch_script(root: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        let desktop = directories::UserDirs::new()
            .and_then(|d| d.desktop_dir().map(Path::to_owned))
            .context("Desktop directory is unavailable")?;
        let path = desktop.join("Start Fatrocu Model.cmd");
        let exe = std::env::current_exe()?;
        ensure!(
            !exe.to_string_lossy().contains(['%', '\r', '\n'])
                && !root.to_string_lossy().contains(['%', '\r', '\n']),
            "Unsupported launch-script path"
        );
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        write!(
            file,
            "@echo off\r\n\"{}\" --data-dir \"{}\" start\r\n",
            exe.display(),
            root.display()
        )?;
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = root;
        anyhow::bail!("Desktop launch scripts are only available on Windows");
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        Config {
            runtime: "runtime".into(),
            model_path: "model".into(),
            projector_path: "projector".into(),
            model_id: DEFAULT_MODEL.into(),
            model_revision: "test".into(),
            runtime_tag: RUNTIME_TAG.into(),
            host: "127.0.0.1".parse().unwrap(),
            port: 1842,
            network_exposure_authorized: false,
            api_key_env: None,
            context_size: 32768,
            gpu_layers: 0,
        }
    }
    #[test]
    fn local_default_and_projector() {
        let c = config();
        assert!(c.validate().is_ok());
        assert!(
            c.arguments()
                .windows(2)
                .any(|w| w == ["--host", "127.0.0.1"])
        );
        assert!(c.arguments().contains(&"--mmproj".into()));
    }
    #[test]
    fn hosting_requires_explicit_authorization_and_key() {
        let mut c = config();
        c.host = "0.0.0.0".parse().unwrap();
        assert!(c.validate().is_err());
        c.network_exposure_authorized = true;
        assert!(c.validate().is_err());
        c.api_key_env = Some("MODEL_API_KEY".into());
        assert!(c.validate().is_ok());
    }
    #[test]
    #[ignore = "Downloads a public runtime artifact; run explicitly during platform integration"]
    fn verified_runtime_archive_starts() {
        let root = tempfile::tempdir().unwrap();
        let binary = install_runtime(root.path(), RUNTIME_TAG).unwrap();
        let output = Command::new(binary).arg("--version").output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
