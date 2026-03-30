// src-tauri/src/lib.rs
//
// Tauri 命令：POS 标注 / 完整分析 / 语料库统计
// 底层通过持久化 Python 子进程调用 TiBERT 模型

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tauri::State;
use tokio::time::sleep;

// ── Types ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenResponse {
    token: String,
    pos: String,
    #[serde(rename = "pos_zh")]
    pos_zh: String,
    #[serde(rename = "is_case_particle")]
    is_case_particle: bool,
    #[serde(rename = "case_name")]
    case_name: Option<String>,
    #[serde(rename = "case_desc")]
    case_desc: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Stats {
    nouns: i32,
    verbs: i32,
    #[serde(rename = "case_particles")]
    case_particles: i32,
    #[serde(rename = "syllable_count")]
    syllable_count: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PosResponse {
    original: String,
    syllables: String,
    tokens: Vec<TokenResponse>,
    stats: Stats,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyzeResponse {
    original: String,
    syllables: String,
    tokens: Vec<TokenResponse>,
    stats: Stats,
    #[serde(rename = "llm_explanation")]
    llm_explanation: Option<String>,
    structure: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorpusStats {
    #[serde(rename = "total_sentences")]
    total_sentences: i32,
    #[serde(rename = "total_collections")]
    total_collections: i32,
    collections: Vec<serde_json::Value>,
    #[serde(rename = "pos_dataset_stats")]
    pos_dataset_stats: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LookupEntry {
    #[serde(rename = "dict_name")]
    dict_name: String,
    definition: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LookupResponse {
    word: String,
    entries: Vec<LookupEntry>,
    #[serde(rename = "verb_entries")]
    verb_entries: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerRequest {
    cmd: String,
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerResponse {
    ok: bool,
    data: Option<serde_json::Value>,
    error: Option<String>,
}

// ── Python Worker Manager ────────────────────────────────────────────────────────

struct PythonWorker {
    child: Child,
}

impl PythonWorker {
    fn new() -> Result<Self> {
        // 找到 Python 虚拟环境中的 Python
        let python = std::env::var("VIRTUAL_ENV")
            .map(|v| format!("{v}/bin/python"))
            .unwrap_or_else(|_| "python".to_string());

        // worker 脚本路径：项目根目录 / src / api / worker.py
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let worker_script = root.join("src/api/worker.py");

        log::info!("启动 Python worker: {} {}", python, worker_script.display());

        let mut child = std::process::Command::new(&python)
            .arg(worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHONUNBUFFERED", "1")
            .spawn()?;

        // 等待 worker 启动
        let _ = child.stderr.take();

        Ok(Self { child })
    }

    fn send(&mut self, req: &WorkerRequest) -> Result<WorkerResponse> {
        let stdin = self.child.stdin.as_mut().unwrap();
        let stdout = self.child.stdout.as_mut().unwrap();

        stdin.write_all((serde_json::to_string(req)? + "\n").as_bytes())?;
        stdin.flush()?;

        let mut line = String::new();
        let mut reader = std::io::BufReader::new(stdout);
        reader.read_line(&mut line)?;

        let resp: WorkerResponse = serde_json::from_str(&line)?;
        Ok(resp)
    }
}

impl Drop for PythonWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

// ── Global State ────────────────────────────────────────────────────────────────

struct AppState {
    worker: Mutex<Option<PythonWorker>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            worker: Mutex::new(None),
        }
    }

    fn get_worker(&self) -> Result<std::sync::MutexGuard<Option<PythonWorker>>> {
        Ok(self.worker.lock().unwrap())
    }
}

// ── Tauri Commands ─────────────────────────────────────────────────────────────

#[tauri::command]
async fn check_health(state: State<'_, AppState>) -> Result<bool, String> {
    let guard = state.get_worker().map_err(|e| e.to_string())?;

    if guard.is_some() {
        // 尝试发送 health 命令
        let req = WorkerRequest {
            cmd: "health".to_string(),
            text: None,
        };
        // drop guard first to avoid deadlock
        drop(guard);
        let mut guard2 = state.worker.lock().unwrap();
        if let Some(ref mut w) = *guard2 {
            match w.send(&req) {
                Ok(r) => return Ok(r.ok),
                Err(_) => return Ok(false),
            }
        }
    }
    Ok(false)
}

#[tauri::command]
async fn pos_tag(text: String, state: State<'_, AppState>) -> Result<PosResponse, String> {
    let mut guard = state.get_worker().map_err(|e| e.to_string())?;

    // 懒启动 worker
    if guard.is_none() {
        match PythonWorker::new() {
            Ok(w) => *guard = Some(w),
            Err(e) => return Err(format!("无法启动 Python worker: {}", e)),
        }
    }

    let req = WorkerRequest {
        cmd: "pos".to_string(),
        text: Some(text),
    };

    // 临时 drop guard 以避免借用在异步块中
    let req_json = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    drop(guard);

    let mut guard2 = state.worker.lock().map_err(|e| e.to_string())?;
    let resp = if let Some(ref mut w) = *guard2 {
        let req: WorkerRequest = serde_json::from_str(&req_json).unwrap();
        w.send(&req).map_err(|e| format!("worker error: {}", e))?
    } else {
        return Err("worker not initialized".to_string());
    };

    if resp.ok {
        serde_json::from_value(resp.data.unwrap())
            .map_err(|e| format!("parse error: {}", e))
    } else {
        Err(resp.error.unwrap_or_else(|| "unknown error".to_string()))
    }
}

#[tauri::command]
async fn analyze(
    text: String,
    use_llm: bool,
    state: State<'_, AppState>,
) -> Result<AnalyzeResponse, String> {
    let mut guard = state.get_worker().map_err(|e| e.to_string())?;

    if guard.is_none() {
        match PythonWorker::new() {
            Ok(w) => *guard = Some(w),
            Err(e) => return Err(format!("无法启动 Python worker: {}", e)),
        }
    }

    #[derive(Serialize)]
    struct AnalyzeReq {
        cmd: String,
        text: Option<String>,
        use_llm: bool,
    }

    let req = AnalyzeReq {
        cmd: "analyze".to_string(),
        text: Some(text),
        use_llm,
    };
    let req_json = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    drop(guard);

    let mut guard2 = state.worker.lock().map_err(|e| e.to_string())?;
    let resp: WorkerResponse = if let Some(ref mut w) = *guard2 {
        // 通过 stdin 发送（通用 JSON）
        let stdin = w.child.stdin.as_mut().unwrap();
        stdin
            .write_all((&req_json + "\n").as_bytes())
            .map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())?;

        let stdout = w.child.stdout.as_mut().unwrap();
        let mut line = String::new();
        let mut reader = std::io::BufReader::new(stdout);
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        serde_json::from_str(&line).map_err(|e| e.to_string())?
    } else {
        return Err("worker not initialized".to_string());
    };

    if resp.ok {
        serde_json::from_value(resp.data.unwrap())
            .map_err(|e| format!("parse error: {}", e))
    } else {
        Err(resp.error.unwrap_or_else(|| "unknown error".to_string()))
    }
}

#[tauri::command]
async fn get_corpus_stats(state: State<'_, AppState>) -> Result<CorpusStats, String> {
    let mut guard = state.get_worker().map_err(|e| e.to_string())?;

    if guard.is_none() {
        match PythonWorker::new() {
            Ok(w) => *guard = Some(w),
            Err(e) => return Err(format!("无法启动 Python worker: {}", e)),
        }
    }

    let req = WorkerRequest {
        cmd: "corpus_stats".to_string(),
        text: None,
    };
    let req_json = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    drop(guard);

    let mut guard2 = state.worker.lock().map_err(|e| e.to_string())?;
    let resp = if let Some(ref mut w) = *guard2 {
        w.send(&req).map_err(|e| format!("worker error: {}", e))?
    } else {
        return Err("worker not initialized".to_string());
    };

    if resp.ok {
        serde_json::from_value(resp.data.unwrap())
            .map_err(|e| format!("parse error: {}", e))
    } else {
        Err(resp.error.unwrap_or_else(|| "unknown error".to_string()))
    }
}

#[tauri::command]
async fn lookup(
    word: String,
    dict_names: Option<Vec<String>>,
    include_verbs: bool,
    state: State<'_, AppState>,
) -> Result<LookupResponse, String> {
    let mut guard = state.get_worker().map_err(|e| e.to_string())?;

    if guard.is_none() {
        match PythonWorker::new() {
            Ok(w) => *guard = Some(w),
            Err(e) => return Err(format!("无法启动 Python worker: {}", e)),
        }
    }

    #[derive(Serialize)]
    struct LookupReq {
        cmd: String,
        word: String,
        dict_names: Option<Vec<String>>,
        include_verbs: bool,
    }

    let req = LookupReq {
        cmd: "lookup".to_string(),
        word,
        dict_names,
        include_verbs,
    };
    let req_json = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    drop(guard);

    let mut guard2 = state.worker.lock().map_err(|e| e.to_string())?;
    let resp: WorkerResponse = if let Some(ref mut w) = *guard2 {
        let stdin = w.child.stdin.as_mut().unwrap();
        stdin
            .write_all((&req_json + "\n").as_bytes())
            .map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())?;

        let stdout = w.child.stdout.as_mut().unwrap();
        let mut line = String::new();
        let mut reader = std::io::BufReader::new(stdout);
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        serde_json::from_str(&line).map_err(|e| e.to_string())?
    } else {
        return Err("worker not initialized".to_string());
    };

    if resp.ok {
        serde_json::from_value(resp.data.unwrap())
            .map_err(|e| format!("parse error: {}", e))
    } else {
        Err(resp.error.unwrap_or_else(|| "unknown error".to_string()))
    }
}

// ── App Entry ──────────────────────────────────────────────────────────────────

pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| {
            use std::io::Write;
            writeln!(
                buf,
                "[{}] {} — {}",
                record.level(),
                record.target(),
                record.args()
            )
        })
        .init();

    log::info!("启动 TiBERT Classical Tauri 应用");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            check_health,
            pos_tag,
            analyze,
            get_corpus_stats,
            lookup,
        ])
        .run(tauri::generate_context!())
        .expect("tauri 应用启动失败");
}
