//! quantus-iniaku — Quantus (QTC) pool miner for iniaku-miner.
//!
//! Speaks the LuckyPool Quantus stratum dialect (JSON-RPC over TCP: login / job / submit,
//! 4-byte extranonce at the top of the 64-byte nonce) and mines with the official
//! Quantus engines (CPU, wgpu, CUDA). An XMRig-style HTTP API (/2/summary, /json_rpc
//! pause/resume) lets iniaku-miner run it on the GPU between Epic ProgPoW turns.
//!
//! Fee: 2% of the mining time (1 minute after 49 minutes), never while paused.
//!
//! Licensed under the Apache License 2.0, like the official quantus-miner it builds on.

use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use engine_cpu::{EngineStatus, JobIdCancelCheck, MinerEngine, Range};
use primitive_types::U512;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const FEE_WALLET: &str = "qzpcmexVVvSeU3d3HgpeCekTBCfv4Fpaqa2tvE63EUzjnBVj8";
const FEE_SLICE: Duration = Duration::from_secs(60);
const FEE_EVERY: Duration = Duration::from_secs(49 * 60);

#[derive(Parser, Debug, Clone)]
#[command(name = "quantus-iniaku", version, about = "Quantus pool miner (LuckyPool stratum) for iniaku-miner")]
struct Args {
	/// Pool host:port (LuckyPool stratum dialect), e.g. sg.lproute.com:5660
	#[arg(long, short = 'o')]
	pool: String,
	/// Login: <QTC address>.<worker>
	#[arg(long, short = 'u')]
	user: String,
	/// Password
	#[arg(long, short = 'p', default_value = "x")]
	pass: String,
	/// CPU worker threads (default 0 = GPU only)
	#[arg(long, default_value_t = 0)]
	cpu_workers: usize,
	/// GPU devices (default: all detected)
	#[arg(long)]
	gpu_devices: Option<usize>,
	/// Native CUDA engine instead of wgpu/Vulkan (NVIDIA)
	#[arg(long)]
	cuda_gpu: bool,
	/// GPU batch size in nonces (default 1000000, 32000000 with --cuda-gpu)
	#[arg(long)]
	gpu_batch_size: Option<u32>,
	/// Allow integrated GPUs
	#[arg(long)]
	allow_integrated: bool,
	/// HTTP API port on 127.0.0.1 (0 = off)
	#[arg(long, default_value_t = 0)]
	api_port: u16,
	/// Bearer token for the HTTP API (required for pause/resume)
	#[arg(long, default_value = "")]
	api_token: String,
	/// Start paused (iniaku-miner resumes it outside the Epic ProgPoW turn)
	#[arg(long)]
	paused: bool,
}

// ---------------------------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Job {
	job_id: String,
	header: [u8; 32],
	difficulty: U512,
	extranonce: [u8; 4],
	session_id: String,
	fee: bool,
}

struct State {
	paused: AtomicBool,
	/// Bumped on every new job / pause: workers drop their current search.
	generation: AtomicU64,
	job: Mutex<Option<Job>>,
	hashes: AtomicU64,
	accepted: AtomicU64,
	rejected: AtomicU64,
	pool: Mutex<String>,
	samples: Mutex<Vec<(Instant, u64)>>,
	started: Instant,
	/// Milliseconds mined for the user (not paused, job present) since the last fee slice.
	mined_ms: AtomicU64,
}

impl State {
	fn hashrate(&self, window: Duration) -> Option<f64> {
		let samples = self.samples.lock().unwrap();
		let (now, last) = *samples.last()?;
		let first = samples.iter().find(|(t, _)| now.duration_since(*t) <= window)?;
		let secs = now.duration_since(first.0).as_secs_f64();
		(secs >= 1.0).then(|| (last - first.1) as f64 / secs)
	}
}

// ---------------------------------------------------------------------------------------------
// Workers: official engines, nonce = extranonce (4 bytes) + random, keep searching after shares
// ---------------------------------------------------------------------------------------------

struct Share {
	job_id: String,
	session_id: String,
	fee: bool,
	nonce: [u8; 64],
}

fn random_nonce(prefix: [u8; 4]) -> U512 {
	let mut bytes = [0u8; 64];
	getrandom::getrandom(&mut bytes).expect("random");
	bytes[..4].copy_from_slice(&prefix);
	// leave room below 2^480 boundary so a chunk never runs into the next prefix
	bytes[4] &= 0x7f;
	U512::from_big_endian(&bytes)
}

fn worker(engine: Arc<dyn MinerEngine>, chunk: u64, state: Arc<State>, shares: Sender<Share>) {
	loop {
		let generation = state.generation.load(Ordering::SeqCst);
		let job = state.job.lock().unwrap().clone();
		let job = match job {
			Some(j) if !state.paused.load(Ordering::SeqCst) => j,
			_ => {
				thread::sleep(Duration::from_millis(50));
				continue;
			}
		};

		let ctx = engine.prepare_context(job.header, job.difficulty);
		let cancel = JobIdCancelCheck { current_job_id: &state.generation, my_job_id: generation };
		let mut start = random_nonce(job.extranonce);

		while state.generation.load(Ordering::SeqCst) == generation {
			let end = start.saturating_add(U512::from(chunk - 1));
			match engine.search_range(&ctx, Range { start, end }, &cancel) {
				EngineStatus::Found { candidate, hash_count, .. } => {
					state.hashes.fetch_add(hash_count, Ordering::Relaxed);
					let _ = shares.try_send(Share {
						job_id: job.job_id.clone(),
						session_id: job.session_id.clone(),
						fee: job.fee,
						nonce: candidate.work,
					});
					start = candidate.nonce.saturating_add(U512::one());
				}
				EngineStatus::Exhausted { hash_count } => {
					state.hashes.fetch_add(hash_count, Ordering::Relaxed);
					start = end.saturating_add(U512::one());
				}
				EngineStatus::Cancelled { hash_count } | EngineStatus::Running { hash_count } => {
					state.hashes.fetch_add(hash_count, Ordering::Relaxed);
					break;
				}
				EngineStatus::DeviceLost { hash_count } => {
					state.hashes.fetch_add(hash_count, Ordering::Relaxed);
					log::error!("GPU device lost, worker stopped");
					return;
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------------------------
// Stratum (LuckyPool Quantus dialect)
// ---------------------------------------------------------------------------------------------

fn parse_job(state: &State, job: &Value, session_id: &str, extranonce: [u8; 4], fee: bool) -> bool {
	let header = job["mining_hash"].as_str().and_then(|h| hex::decode(h.trim_start_matches("0x")).ok());
	let job_id = job["job_id"].as_str().map(str::to_string).or_else(|| job["job_id"].as_u64().map(|n| n.to_string()));
	let difficulty = match job.get("difficulty") {
		Some(Value::Number(n)) => n.as_u64().map(U512::from).or_else(|| n.as_f64().map(|f| U512::from(f as u64))),
		Some(Value::String(s)) => U512::from_dec_str(s).ok(),
		_ => None,
	}
	.or_else(|| {
		let t = U512::from_str_radix(job["target"].as_str()?, 16).ok()?;
		(!t.is_zero()).then(|| U512::MAX / t)
	});

	match (header, job_id, difficulty) {
		(Some(h), Some(id), Some(d)) if h.len() == 32 && !d.is_zero() => {
			let mut header = [0u8; 32];
			header.copy_from_slice(&h);
			*state.job.lock().unwrap() = Some(Job {
				job_id: id.clone(),
				header,
				difficulty: d,
				extranonce,
				session_id: session_id.to_string(),
				fee,
			});
			state.generation.fetch_add(1, Ordering::SeqCst);
			log::info!("new job {} diff {}{}", id, d, if fee { " (fee)" } else { "" });
			true
		}
		_ => {
			log::warn!("invalid job: {job}");
			false
		}
	}
}

/// One pool session. Returns when the connection drops or `until` (fee switch) is reached.
fn session(args: &Args, login: &str, fee: bool, state: &Arc<State>, shares: &Receiver<Share>, leave: &dyn Fn() -> bool) -> anyhow::Result<()> {
	let addr = args.pool.trim_start_matches("stratum+tcp://").to_string();
	let sock = addr
		.to_socket_addrs()?
		.next()
		.ok_or_else(|| anyhow::anyhow!("cannot resolve {addr}"))?;
	let stream = TcpStream::connect_timeout(&sock, Duration::from_secs(10))?;
	stream.set_read_timeout(Some(Duration::from_millis(200)))?;
	let mut writer = stream.try_clone()?;
	let mut reader = BufReader::new(stream);
	*state.pool.lock().unwrap() = addr.clone();

	let send = |w: &mut TcpStream, v: Value| -> std::io::Result<()> {
		let mut line = v.to_string();
		line.push('\n');
		w.write_all(line.as_bytes())
	};

	send(&mut writer, json!({"id": 1, "jsonrpc": "2.0", "method": "login",
		"params": {"login": login, "pass": args.pass, "agent": format!("quantus-iniaku/{VERSION}")}}))?;
	log::info!("{} {} as {}", if fee { "fee: connecting to" } else { "connecting to" }, addr, short(login));

	let mut session_id = String::new();
	let mut extranonce = [0u8; 4];
	let mut next_id = 10u64;
	let mut pending: std::collections::HashMap<u64, bool> = Default::default();
	let mut last_keepalive = Instant::now();
	let mut line = String::new();

	loop {
		if leave() {
			return Ok(());
		}

		// submit found shares
		while let Ok(share) = shares.try_recv() {
			if share.session_id != session_id || share.fee != fee {
				continue; // from an earlier session / login
			}
			next_id += 1;
			pending.insert(next_id, true);
			send(&mut writer, json!({"id": next_id, "jsonrpc": "2.0", "method": "submit",
				"params": {"id": session_id, "job_id": share.job_id, "nonce": hex::encode(share.nonce)}}))?;
		}

		if last_keepalive.elapsed() > Duration::from_secs(60) {
			last_keepalive = Instant::now();
			next_id += 1;
			send(&mut writer, json!({"id": next_id, "jsonrpc": "2.0", "method": "keepalived", "params": {"id": session_id}}))?;
		}

		line.clear();
		match reader.read_line(&mut line) {
			Ok(0) => anyhow::bail!("pool closed the connection"),
			Ok(_) => {}
			Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => continue,
			Err(e) => return Err(e.into()),
		}
		let msg: Value = match serde_json::from_str(line.trim()) {
			Ok(v) => v,
			Err(_) => continue,
		};

		if msg.get("method").and_then(Value::as_str) == Some("job") {
			let params = &msg["params"];
			let job = if params.get("job").is_some() { &params["job"] } else { params };
			if let Some(x) = job["extranonce"].as_str().and_then(|x| hex::decode(x).ok()).filter(|x| x.len() == 4) {
				extranonce.copy_from_slice(&x);
			}
			parse_job(state, job, &session_id, extranonce, fee);
			continue;
		}

		let id = msg["id"].as_u64().unwrap_or(0);
		if id == 1 {
			if !msg["error"].is_null() {
				anyhow::bail!("login failed: {}", msg["error"]);
			}
			let result = &msg["result"];
			session_id = result["id"].as_str().unwrap_or("").to_string();
			if let Some(x) = hex::decode(&session_id).ok().filter(|x| x.len() == 4) {
				extranonce.copy_from_slice(&x);
			}
			if let Some(x) = result["job"]["extranonce"].as_str().and_then(|x| hex::decode(x).ok()).filter(|x| x.len() == 4) {
				extranonce.copy_from_slice(&x);
			}
			log::info!("logged in, session {session_id}");
			parse_job(state, &result["job"], &session_id, extranonce, fee);
		} else if pending.remove(&id).is_some() {
			let ok = msg["error"].is_null();
			if !fee {
				if ok {
					state.accepted.fetch_add(1, Ordering::Relaxed);
				} else {
					state.rejected.fetch_add(1, Ordering::Relaxed);
				}
			}
			let (a, r) = (state.accepted.load(Ordering::Relaxed), state.rejected.load(Ordering::Relaxed));
			if ok {
				log::info!("{}accepted ({}/{})", if fee { "fee " } else { "" }, a, r);
			} else {
				log::warn!("{}rejected ({}/{}) {}", if fee { "fee " } else { "" }, a, r, msg["error"]);
			}
		}
	}
}

fn short(login: &str) -> String {
	if login.len() > 20 {
		format!("{}…{}", &login[..8], &login[login.len() - 10..])
	} else {
		login.to_string()
	}
}

// ---------------------------------------------------------------------------------------------
// HTTP API (XMRig compatible subset)
// ---------------------------------------------------------------------------------------------

fn api(port: u16, token: String, state: Arc<State>) {
	let listener = match TcpListener::bind(("127.0.0.1", port)) {
		Ok(l) => l,
		Err(e) => {
			log::error!("API port {port}: {e}");
			return;
		}
	};
	log::info!("HTTP API on 127.0.0.1:{port}");
	for stream in listener.incoming().flatten() {
		let state = state.clone();
		let token = token.clone();
		thread::spawn(move || {
			let _ = handle_api(stream, &token, &state);
		});
	}
}

fn handle_api(mut stream: TcpStream, token: &str, state: &State) -> std::io::Result<()> {
	stream.set_read_timeout(Some(Duration::from_secs(3)))?;
	let mut buf = vec![0u8; 8192];
	let mut n = 0;
	loop {
		let r = stream.read(&mut buf[n..])?;
		if r == 0 {
			break;
		}
		n += r;
		let text = String::from_utf8_lossy(&buf[..n]);
		if let Some(h) = text.find("\r\n\r\n") {
			let len = text[..h]
				.lines()
				.find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
				.unwrap_or(0);
			if n >= h + 4 + len || n == buf.len() {
				break;
			}
		}
	}
	let text = String::from_utf8_lossy(&buf[..n]).to_string();
	let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
	let request_line = head.lines().next().unwrap_or("");
	let authorized = token.is_empty() || head.lines().any(|l| l.trim() == format!("Authorization: Bearer {token}"));

	let (status, reply) = if request_line.starts_with("GET /2/summary") || request_line.starts_with("GET /1/summary") {
		let total = [state.hashrate(Duration::from_secs(10)), state.hashrate(Duration::from_secs(60)), state.hashrate(Duration::from_secs(900))];
		let (a, r) = (state.accepted.load(Ordering::Relaxed), state.rejected.load(Ordering::Relaxed));
		("200 OK", json!({
			"version": VERSION, "uptime": state.started.elapsed().as_secs(),
			"paused": state.paused.load(Ordering::Relaxed),
			"hashrate": {"total": total},
			"results": {"shares_good": a, "shares_total": a + r},
			"connection": {"pool": *state.pool.lock().unwrap()},
		}))
	} else if request_line.starts_with("POST /json_rpc") {
		if !authorized {
			("401 Unauthorized", json!({"error": "UNAUTHORIZED"}))
		} else {
			let req: Value = serde_json::from_str(body).unwrap_or(Value::Null);
			let method = req["method"].as_str().unwrap_or("");
			match method {
				"pause" | "resume" => {
					let pause = method == "pause";
					if state.paused.swap(pause, Ordering::SeqCst) != pause {
						state.generation.fetch_add(1, Ordering::SeqCst);
						log::info!("{}", if pause { "paused" } else { "resumed" });
					}
					("200 OK", json!({"id": req["id"], "jsonrpc": "2.0", "result": {"status": "OK"}}))
				}
				_ => ("200 OK", json!({"id": req["id"], "jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}})),
			}
		}
	} else {
		("404 Not Found", json!({"error": "NOT_FOUND"}))
	};

	let body = serde_json::to_string_pretty(&reply).unwrap();
	write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}", body.len(), body)
}

// ---------------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
	env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,cuda_engine=warn,engine_gpu=warn,engine_cuda=warn"))
		.format_timestamp_millis()
		.init();
	let args = Args::parse();
	log::info!("quantus-iniaku {VERSION} (official Quantus engines, LuckyPool stratum) — 2% fee");

	let gpu_batch = args.gpu_batch_size.unwrap_or(if args.cuda_gpu { 32_000_000 } else { 1_000_000 });
	let (gpu_engine, gpu_devices) = if args.cuda_gpu {
		miner_service::resolve_cuda_configuration(args.gpu_devices, gpu_batch, 0)?
	} else {
		miner_service::resolve_gpu_configuration(args.gpu_devices, gpu_batch, 0, args.allow_integrated)?
	};
	if args.cpu_workers == 0 && gpu_devices == 0 {
		anyhow::bail!("no GPU found and --cpu-workers is 0");
	}
	log::info!("{} CPU worker(s), {} GPU device(s){}", args.cpu_workers, gpu_devices,
		gpu_engine.as_ref().map(|e| format!(" ({})", e.name())).unwrap_or_default());

	let state = Arc::new(State {
		paused: AtomicBool::new(args.paused),
		generation: AtomicU64::new(0),
		job: Mutex::new(None),
		hashes: AtomicU64::new(0),
		accepted: AtomicU64::new(0),
		rejected: AtomicU64::new(0),
		pool: Mutex::new(String::new()),
		samples: Mutex::new(vec![]),
		started: Instant::now(),
		mined_ms: AtomicU64::new(0),
	});
	let (share_tx, share_rx) = bounded::<Share>(1024);

	let cpu_engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::FastCpuEngine::new(10_000));
	for _ in 0..args.cpu_workers {
		let (e, s, t) = (cpu_engine.clone(), state.clone(), share_tx.clone());
		// ~1 s per chunk so the hashrate updates smoothly and pauses are quick
		thread::spawn(move || worker(e, 1 << 16, s, t));
	}
	if let Some(engine) = gpu_engine {
		for _ in 0..gpu_devices {
			let (e, s, t) = (engine.clone(), state.clone(), share_tx.clone());
			thread::spawn(move || worker(e, (gpu_batch as u64) * 4, s, t));
		}
	}

	if args.api_port > 0 {
		let (p, t, s) = (args.api_port, args.api_token.clone(), state.clone());
		thread::spawn(move || api(p, t, s));
	}

	// hashrate samples + periodic speed line
	{
		let state = state.clone();
		thread::spawn(move || {
			let mut last_print = Instant::now();
			loop {
				thread::sleep(Duration::from_secs(1));
				if !state.paused.load(Ordering::Relaxed) && state.job.lock().unwrap().as_ref().map(|j| !j.fee).unwrap_or(false) {
					state.mined_ms.fetch_add(1000, Ordering::Relaxed);
				}
				{
					let mut s = state.samples.lock().unwrap();
					s.push((Instant::now(), state.hashes.load(Ordering::Relaxed)));
					let len = s.len();
					if len > 1000 {
						s.drain(..len - 1000);
					}
				}
				if last_print.elapsed() >= Duration::from_secs(30) {
					last_print = Instant::now();
					let fmt = |h: Option<f64>| h.map(pow_core::format_hashrate).unwrap_or_else(|| "n/a".into());
					log::info!("speed 10s/60s {} {}{}", fmt(state.hashrate(Duration::from_secs(10))),
						fmt(state.hashrate(Duration::from_secs(60))), if state.paused.load(Ordering::Relaxed) { " (paused)" } else { "" });
				}
			}
		});
	}

	// pool session; leaves for a 1 minute fee slice after 49 minutes of (unpaused) mining
	let fee_login = format!("{FEE_WALLET}.iniaku-fee");
	let fee_due = || state.mined_ms.load(Ordering::Relaxed) >= FEE_EVERY.as_millis() as u64 && !state.paused.load(Ordering::Relaxed);
	loop {
		if let Err(e) = session(&args, &args.user, false, &state, &share_rx, &fee_due) {
			log::warn!("pool: {e}, reconnecting in 10 s");
			*state.job.lock().unwrap() = None;
			state.generation.fetch_add(1, Ordering::SeqCst);
			thread::sleep(Duration::from_secs(10));
			continue;
		}
		state.mined_ms.fetch_sub(FEE_EVERY.as_millis() as u64, Ordering::Relaxed);
		log::info!("fee 2% started (60 s)");
		// the fee slice only counts down while mining (a pause stretches it)
		let mut left = FEE_SLICE;
		let mut tick = Instant::now();
		let fee_done = || {
			let now = Instant::now();
			if !state.paused.load(Ordering::Relaxed) {
				left = left.saturating_sub(now.duration_since(tick));
			}
			tick = now;
			left.is_zero()
		};
		let fee_done = std::cell::RefCell::new(fee_done);
		if let Err(e) = session(&args, &fee_login, true, &state, &share_rx, &|| (fee_done.borrow_mut())()) {
			log::warn!("fee pool: {e}");
		}
		log::info!("fee finished");
		*state.job.lock().unwrap() = None;
		state.generation.fetch_add(1, Ordering::SeqCst);
	}
}
