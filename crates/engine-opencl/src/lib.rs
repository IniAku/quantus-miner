#![deny(rust_2018_idioms)]
//! OpenCL mining engine (AMD and other OpenCL GPUs).
//!
//! Same kernel and search contract as `engine-cuda` (see its docs): every
//! launch evaluates its whole nonce rectangle, up to `MAX_HITS` candidates are
//! recorded, and the host re-verifies each candidate with the exact CPU hash.
//! OpenCL is loaded at runtime (`cl.rs`), so no SDK is needed to build.
//!
//! Tuning (environment): `QUANTUS_OPENCL_LOCAL` work-group size (default 256),
//! `QUANTUS_OPENCL_BLOCKS` max work-groups per launch (default 4096),
//! `QUANTUS_OPENCL_ANY_DEVICE=1` also use non-GPU devices (tests on pocl).

mod cl;

use cl::*;
use engine_cpu::{CancelCheck, Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use pow_core::{format_hashrate, format_u512, JobContext};
use primitive_types::U512;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("kernels/mining.cl");
const MAX_HITS: usize = 8;
const RESULTS_U32S: usize = 1 + MAX_HITS;
const PARAMS_U32S: usize = 24 + 16 + 16 + 3;

struct ClDevice {
    device: cl_device_id,
    context: cl_context,
    program: cl_program,
    name: String,
}

// OpenCL contexts, programs and device ids are thread-safe (OpenCL 1.2 §A.2);
// kernels and queues are created per worker thread.
unsafe impl Send for ClDevice {}
unsafe impl Sync for ClDevice {}

struct WorkerBuffers {
    engine_id: usize,
    device_index: usize,
    queue: cl_command_queue,
    results: cl_mem,
    params: cl_mem,
    mine: cl_kernel,
}

impl Drop for WorkerBuffers {
    fn drop(&mut self) {
        if let Ok(api) = api() {
            unsafe {
                (api.clReleaseKernel)(self.mine);
                (api.clReleaseMemObject)(self.results);
                (api.clReleaseMemObject)(self.params);
                (api.clReleaseCommandQueue)(self.queue);
            }
        }
    }
}

pub struct OpenClEngine {
    engine_id: usize,
    devices: Vec<Arc<ClDevice>>,
    device_counter: AtomicUsize,
    batch_size: u32,
    throttle_ms: u64,
    local_size: u32,
    max_groups: u32,
}

static ENGINE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ASSIGNED_DEVICE: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
    static WORKER_BUFFERS: RefCell<Option<WorkerBuffers>> = const { RefCell::new(None) };
    static DEVICE_LOST: RefCell<Option<usize>> = const { RefCell::new(None) };
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn device_info_string(api: &Api, dev: cl_device_id, param: cl_uint) -> String {
    info_string(api.clGetDeviceInfo, dev, param)
}

fn build_program(api: &Api, context: cl_context, dev: cl_device_id, name: &str) -> Result<cl_program, String> {
    let mut err = 0;
    let src = KERNEL_SRC.as_ptr();
    let len = KERNEL_SRC.len();
    let program = unsafe { (api.clCreateProgramWithSource)(context, 1, &src, &len, &mut err) };
    check(err, "clCreateProgramWithSource")?;
    let opts = b"-cl-std=CL1.2\0";
    let code = unsafe { (api.clBuildProgram)(program, 1, &dev, opts.as_ptr(), std::ptr::null(), std::ptr::null_mut()) };
    if code != CL_SUCCESS {
        let mut size = 0usize;
        let mut log_text = String::new();
        unsafe {
            if (api.clGetProgramBuildInfo)(program, dev, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut size) == CL_SUCCESS {
                let mut buf = vec![0u8; size.max(1)];
                (api.clGetProgramBuildInfo)(program, dev, CL_PROGRAM_BUILD_LOG, size, buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
                log_text = String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string();
            }
        }
        return Err(format!("OpenCL kernel build failed on {name} (error {code}): {log_text}"));
    }
    Ok(program)
}

impl OpenClEngine {
    pub fn try_new(batch_size: u32, throttle_ms: u64) -> Result<Self, Box<dyn std::error::Error>> {
        if batch_size == 0 {
            return Err("batch_size must be non-zero".into());
        }
        let api = api()?;
        let any_device = std::env::var("QUANTUS_OPENCL_ANY_DEVICE").is_ok_and(|v| v == "1");
        let dev_type = if any_device { CL_DEVICE_TYPE_ALL } else { CL_DEVICE_TYPE_GPU };

        let mut n = 0u32;
        unsafe { (api.clGetPlatformIDs)(0, std::ptr::null_mut(), &mut n) };
        if n == 0 {
            return Err("No OpenCL platforms found".into());
        }
        let mut platforms = vec![std::ptr::null_mut(); n as usize];
        check(unsafe { (api.clGetPlatformIDs)(n, platforms.as_mut_ptr(), std::ptr::null_mut()) }, "clGetPlatformIDs")?;

        let mut devices = Vec::new();
        for platform in platforms {
            let pname = info_string(api.clGetPlatformInfo, platform, CL_PLATFORM_NAME);
            let mut dn = 0u32;
            if unsafe { (api.clGetDeviceIDs)(platform, dev_type, 0, std::ptr::null_mut(), &mut dn) } != CL_SUCCESS || dn == 0 {
                continue;
            }
            let mut ids = vec![std::ptr::null_mut(); dn as usize];
            if unsafe { (api.clGetDeviceIDs)(platform, dev_type, dn, ids.as_mut_ptr(), std::ptr::null_mut()) } != CL_SUCCESS {
                continue;
            }
            for dev in ids {
                let name = device_info_string(api, dev, CL_DEVICE_NAME);
                let vendor = device_info_string(api, dev, CL_DEVICE_VENDOR);
                let mut cus: cl_uint = 0;
                unsafe {
                    (api.clGetDeviceInfo)(dev, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &mut cus as *mut _ as *mut c_void, std::ptr::null_mut());
                }
                let mut err = 0;
                let context = unsafe { (api.clCreateContext)(std::ptr::null(), 1, &dev, std::ptr::null(), std::ptr::null_mut(), &mut err) };
                if err != CL_SUCCESS {
                    log::warn!(target: "opencl_engine", "Skipping {name}: clCreateContext error {err}");
                    continue;
                }
                let started = Instant::now();
                let program = match build_program(api, context, dev, &name) {
                    Ok(p) => p,
                    Err(e) => {
                        log::warn!(target: "opencl_engine", "Skipping {name}: {e}");
                        continue;
                    }
                };
                log::info!(
                    target: "opencl_engine",
                    "OpenCL device {}: {} ({}, {}, {} CUs; kernel built in {:.1}s)",
                    devices.len(), name, vendor, pname, cus, started.elapsed().as_secs_f64()
                );
                devices.push(Arc::new(ClDevice { device: dev, context, program, name }));
            }
        }
        if devices.is_empty() {
            return Err("No usable OpenCL GPU found".into());
        }
        let local_size = env_u32("QUANTUS_OPENCL_LOCAL", 256);
        let max_groups = env_u32("QUANTUS_OPENCL_BLOCKS", 4096);
        log::info!(
            target: "opencl_engine",
            "OpenCL engine initialized with {} device(s) (batch size: {batch_size} nonces, work-group {local_size}, max groups {max_groups})",
            devices.len()
        );
        Ok(Self {
            engine_id: ENGINE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            devices,
            device_counter: AtomicUsize::new(0),
            batch_size,
            throttle_ms,
            local_size,
            max_groups,
        })
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn clear_worker_resources() {
        WORKER_BUFFERS.with(|b| *b.borrow_mut() = None);
        ASSIGNED_DEVICE.with(|a| *a.borrow_mut() = None);
        DEVICE_LOST.with(|l| *l.borrow_mut() = None);
    }

    /// Hashes `count` consecutive nonces on the first device with the kernel's
    /// arithmetic (self-test, not a verifier).
    pub fn hash_nonces(&self, header: [u8; 32], start: U512, count: u32) -> Result<Vec<U512>, Box<dyn std::error::Error>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let api = api()?;
        let dev = &self.devices[0];
        let nonce_be = start.to_big_endian();
        let mid = pow_core::mining_midstate_u32s(header, nonce_be[..32].try_into().unwrap());
        let start_limbs = pow_core::u512_to_le_u32s(start);
        unsafe {
            let mut err = 0;
            let queue = (api.clCreateCommandQueue)(dev.context, dev.device, 0, &mut err);
            check(err, "clCreateCommandQueue")?;
            let kernel = (api.clCreateKernel)(dev.program, b"hash_nonces\0".as_ptr(), &mut err);
            check(err, "clCreateKernel(hash_nonces)")?;
            let out = (api.clCreateBuffer)(dev.context, CL_MEM_READ_WRITE, count as usize * 64, std::ptr::null_mut(), &mut err);
            check(err, "clCreateBuffer")?;
            let mid_buf = (api.clCreateBuffer)(dev.context, CL_MEM_READ_ONLY, 24 * 4, std::ptr::null_mut(), &mut err);
            check(err, "clCreateBuffer")?;
            let start_buf = (api.clCreateBuffer)(dev.context, CL_MEM_READ_ONLY, 16 * 4, std::ptr::null_mut(), &mut err);
            check(err, "clCreateBuffer")?;
            check((api.clEnqueueWriteBuffer)(queue, mid_buf, CL_TRUE, 0, 96, mid.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut()), "write")?;
            check((api.clEnqueueWriteBuffer)(queue, start_buf, CL_TRUE, 0, 64, start_limbs.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut()), "write")?;
            check((api.clSetKernelArg)(kernel, 0, 8, &out as *const _ as *const c_void), "arg0")?;
            check((api.clSetKernelArg)(kernel, 1, 8, &mid_buf as *const _ as *const c_void), "arg1")?;
            check((api.clSetKernelArg)(kernel, 2, 8, &start_buf as *const _ as *const c_void), "arg2")?;
            check((api.clSetKernelArg)(kernel, 3, 4, &count as *const _ as *const c_void), "arg3")?;
            let local = 64usize;
            let global = (count as usize).div_ceil(local) * local;
            check((api.clEnqueueNDRangeKernel)(queue, kernel, 1, std::ptr::null(), &global, &local, 0, std::ptr::null(), std::ptr::null_mut()), "launch")?;
            let mut raw = vec![0u32; count as usize * 16];
            check((api.clEnqueueReadBuffer)(queue, out, CL_TRUE, 0, raw.len() * 4, raw.as_mut_ptr() as *mut c_void, 0, std::ptr::null(), std::ptr::null_mut()), "read")?;
            (api.clReleaseMemObject)(out);
            (api.clReleaseMemObject)(mid_buf);
            (api.clReleaseMemObject)(start_buf);
            (api.clReleaseKernel)(kernel);
            (api.clReleaseCommandQueue)(queue);
            Ok(raw
                .chunks(16)
                .map(|c| pow_core::u512_from_le_u32s(c.try_into().unwrap()))
                .collect())
        }
    }
}

fn create_buffers(engine_id: usize, device_index: usize, dev: &ClDevice) -> Result<WorkerBuffers, String> {
    let api = api()?;
    let mut err = 0;
    unsafe {
        let queue = (api.clCreateCommandQueue)(dev.context, dev.device, 0, &mut err);
        check(err, "clCreateCommandQueue")?;
        let mine = (api.clCreateKernel)(dev.program, b"mining_main\0".as_ptr(), &mut err);
        check(err, "clCreateKernel(mining_main)")?;
        let results = (api.clCreateBuffer)(dev.context, CL_MEM_READ_WRITE, RESULTS_U32S * 4, std::ptr::null_mut(), &mut err);
        check(err, "clCreateBuffer(results)")?;
        let params = (api.clCreateBuffer)(dev.context, CL_MEM_READ_ONLY, PARAMS_U32S * 4, std::ptr::null_mut(), &mut err);
        check(err, "clCreateBuffer(params)")?;
        check((api.clSetKernelArg)(mine, 0, 8, &results as *const _ as *const c_void), "clSetKernelArg(results)")?;
        check((api.clSetKernelArg)(mine, 1, 8, &params as *const _ as *const c_void), "clSetKernelArg(params)")?;
        Ok(WorkerBuffers { engine_id, device_index, queue, results, params, mine })
    }
}

enum BatchResult {
    Found { candidate: Candidate, hash_count: u64 },
    NotFound { hash_count: u64 },
    DeviceLost,
}

fn nonces_until_low64_carry(nonce: U512) -> U512 {
    (U512::one() << 64).saturating_sub(U512::from(nonce.low_u64()))
}

fn run_single_batch(
    engine: &OpenClEngine,
    b: &mut WorkerBuffers,
    ctx: &JobContext,
    batch_start: U512,
    batch_size: u32,
) -> BatchResult {
    let api = match api() {
        Ok(a) => a,
        Err(_) => return BatchResult::DeviceLost,
    };
    let local = engine.local_size;
    let groups = batch_size.div_ceil(local).clamp(1, engine.max_groups);
    let total_threads = groups * local;
    let nonces_per_thread = batch_size.div_ceil(total_threads).max(1);
    let nonce_be = batch_start.to_big_endian();
    let mut params = [0u32; PARAMS_U32S];
    params[..24].copy_from_slice(&pow_core::mining_prestate_low64_u32s(ctx.header, nonce_be));
    params[24..40].copy_from_slice(&pow_core::u512_to_le_u32s(batch_start));
    params[40..56].copy_from_slice(&pow_core::u512_to_le_u32s(ctx.target));
    params[56..59].copy_from_slice(&[total_threads, nonces_per_thread, batch_size]);

    let mut results = [0u32; RESULTS_U32S];
    let run = unsafe {
        let null = std::ptr::null();
        let nullm = std::ptr::null_mut();
        let zero = [0u32; RESULTS_U32S];
        let global = total_threads as usize;
        let local_sz = local as usize;
        check((api.clEnqueueWriteBuffer)(b.queue, b.results, 0, 0, RESULTS_U32S * 4, zero.as_ptr() as *const c_void, 0, null, nullm), "write results")
            .and_then(|_| check((api.clEnqueueWriteBuffer)(b.queue, b.params, 0, 0, PARAMS_U32S * 4, params.as_ptr() as *const c_void, 0, null, nullm), "write params"))
            .and_then(|_| check((api.clEnqueueNDRangeKernel)(b.queue, b.mine, 1, std::ptr::null(), &global, &local_sz, 0, null, nullm), "launch"))
            .and_then(|_| check((api.clEnqueueReadBuffer)(b.queue, b.results, CL_TRUE, 0, RESULTS_U32S * 4, results.as_mut_ptr() as *mut c_void, 0, null, nullm), "read results"))
    };
    if let Err(e) = run {
        log::error!(target: "opencl_engine", "OpenCL batch failed: {e}");
        return BatchResult::DeviceLost;
    }

    let dispatched = (total_threads as u64 * nonces_per_thread as u64).min(batch_size as u64);
    let hits = results[0] as usize;
    if hits > MAX_HITS {
        log::warn!(target: "opencl_engine", "OpenCL launch produced {hits} candidates, only {MAX_HITS} were recorded");
    }
    let mut best: Option<(U512, U512)> = None;
    for &index in &results[1..=hits.min(MAX_HITS)] {
        let logical_index = index as u64;
        if logical_index >= dispatched {
            log::error!(target: "opencl_engine", "OpenCL returned out-of-range candidate index {logical_index}");
            return BatchResult::DeviceLost;
        }
        let nonce = batch_start + U512::from(logical_index);
        let hash = pow_core::hash_from_nonce(ctx, nonce);
        if hash >= ctx.target {
            log::warn!(target: "opencl_engine", "OpenCL candidate {} rejected by CPU verification", format_u512(nonce));
            continue;
        }
        if best.is_none_or(|(n, _)| nonce < n) {
            best = Some((nonce, hash));
        }
    }
    match best {
        Some((nonce, hash)) => BatchResult::Found {
            candidate: Candidate { nonce, work: nonce.to_big_endian(), hash },
            hash_count: dispatched,
        },
        None => BatchResult::NotFound { hash_count: dispatched },
    }
}

impl MinerEngine for OpenClEngine {
    fn name(&self) -> &'static str {
        "gpu-opencl"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &dyn CancelCheck) -> EngineStatus {
        if self.devices.is_empty() || range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }
        if DEVICE_LOST.with(|lost| *lost.borrow() == Some(self.engine_id)) {
            return EngineStatus::DeviceLost { hash_count: 0 };
        }
        if cancel.is_cancelled() {
            return EngineStatus::Cancelled { hash_count: 0 };
        }

        let device_index = ASSIGNED_DEVICE.with(|assigned| {
            let mut a = assigned.borrow_mut();
            match *a {
                Some((engine_id, index)) if engine_id == self.engine_id => index,
                _ => {
                    let index = self.device_counter.fetch_add(1, Ordering::SeqCst) % self.devices.len();
                    *a = Some((self.engine_id, index));
                    log::info!(target: "opencl_engine", "Worker thread assigned to OpenCL device {} ({})", index, self.devices[index].name);
                    index
                }
            }
        });

        let ready = WORKER_BUFFERS.with(|cell| {
            let mut slot = cell.borrow_mut();
            let reuse = matches!(&*slot, Some(b) if b.engine_id == self.engine_id && b.device_index == device_index);
            if !reuse {
                *slot = Some(create_buffers(self.engine_id, device_index, &self.devices[device_index])?);
            }
            Ok::<(), String>(())
        });
        if let Err(e) = ready {
            log::error!(target: "opencl_engine", "OpenCL buffer setup failed: {e}");
            DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
            return EngineStatus::DeviceLost { hash_count: 0 };
        }

        let search_start = Instant::now();
        let mut total_hashes: u64 = 0;
        let mut current_start = range.start;
        let mut batch_num = 0u64;

        while current_start <= range.end {
            if cancel.is_cancelled() {
                return EngineStatus::Cancelled { hash_count: total_hashes };
            }
            let remaining = range.end.saturating_sub(current_start).saturating_add(U512::one());
            let cap = remaining.min(nonces_until_low64_carry(current_start));
            let this_batch: u32 = if cap > U512::from(self.batch_size) { self.batch_size } else { cap.low_u32() };

            let result = WORKER_BUFFERS.with(|cell| {
                let mut slot = cell.borrow_mut();
                run_single_batch(self, slot.as_mut().expect("OpenCL buffers initialized"), ctx, current_start, this_batch)
            });
            match result {
                BatchResult::Found { candidate, hash_count } => {
                    total_hashes += hash_count;
                    return EngineStatus::Found { candidate, hash_count: total_hashes, origin: FoundOrigin::GpuG1 };
                }
                BatchResult::NotFound { hash_count } => total_hashes += hash_count,
                BatchResult::DeviceLost => {
                    DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
                    WORKER_BUFFERS.with(|res| *res.borrow_mut() = None);
                    return EngineStatus::DeviceLost { hash_count: total_hashes };
                }
            }
            current_start = current_start.saturating_add(U512::from(this_batch));
            batch_num += 1;

            if self.throttle_ms > 0 && current_start <= range.end {
                std::thread::sleep(std::time::Duration::from_millis(self.throttle_ms));
            }
            if batch_num.is_multiple_of(10) {
                let elapsed = search_start.elapsed();
                log::debug!(
                    target: "opencl_engine",
                    "OpenCL device {} batch {}: {} hashes ({:.2}s, {})",
                    device_index, batch_num, total_hashes, elapsed.as_secs_f64(),
                    format_hashrate(total_hashes as f64 / elapsed.as_secs_f64())
                );
            }
        }
        EngineStatus::Exhausted { hash_count: total_hashes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Option<OpenClEngine> {
        std::env::set_var("QUANTUS_OPENCL_ANY_DEVICE", "1");
        match OpenClEngine::try_new(1 << 16, 0) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("skipping: {e}");
                None
            }
        }
    }

    /// Kernel hashes equal the exact CPU hash, across a low-word carry.
    #[test]
    fn opencl_hashes_match_cpu() {
        let Some(e) = engine() else { return };
        let header = [0x5au8; 32];
        let ctx = JobContext::new(header, U512::from(1u64));
        let start = (U512::one() << 32) - U512::from(100u64);
        let got = e.hash_nonces(header, start, 256).unwrap();
        for (i, h) in got.iter().enumerate() {
            let want = pow_core::hash_from_nonce(&ctx, start + U512::from(i as u64));
            assert_eq!(*h, want, "nonce index {i}");
        }
    }

    /// mining_main finds a nonce the CPU agrees with, at an easy difficulty.
    #[test]
    fn opencl_search_finds_cpu_verified_solution() {
        let Some(e) = engine() else { return };
        let header = [0x11u8; 32];
        let ctx = JobContext::new(header, U512::from(200_000u64));
        struct Never;
        impl CancelCheck for Never {
            fn is_cancelled(&self) -> bool {
                false
            }
        }
        let start = U512::from(12345u64);
        let range = Range { start, end: start + U512::from(5_000_000u64) };
        match e.search_range(&ctx, range, &Never) {
            EngineStatus::Found { candidate, .. } => {
                assert!(pow_core::hash_from_nonce(&ctx, candidate.nonce) < ctx.target);
                // lowest solution in the first batch must match a CPU scan
                let mut n = start;
                while pow_core::hash_from_nonce(&ctx, n) >= ctx.target {
                    n = n + U512::one();
                }
                assert_eq!(candidate.nonce, n);
            }
            other => panic!("no solution found: {other:?}"),
        }
    }
}
