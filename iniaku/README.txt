quantus-iniaku — Quantus (QTC) pool miner
=========================================

Quantus pool miner used by iniaku-miner (github.com/IniAku/iniaku-miner) for the GPU time
between Epic Cash ProgPoW turns. It can also run on its own:

    quantus-iniaku -o sg.lproute.com:5660 -u <QTC address>.<worker> --gpu-devices 1 --cuda-gpu

  -o, --pool           pool host:port (LuckyPool stratum dialect, plain TCP)
  -u, --user           <QTC address>.<worker>
  --gpu-devices N      number of GPUs (default: all)
  --cuda-gpu           native CUDA engine (NVIDIA); default is wgpu (Vulkan / DX12)
  --cpu-workers N      CPU threads (default 0)
  --api-port P         XMRig-style HTTP API on 127.0.0.1 (/2/summary, /json_rpc pause/resume)
  --api-token T        token required for pause/resume
  --paused             start paused

Fee: 1% of the mining time (1 minute after 99 minutes of mining, never while paused).
The CUDA engine needs the NVIDIA driver; NVRTC (nvrtc64_*.dll / libnvrtc.so.12) must be
next to the miner or on the library path.

Based on the official Quantus miner (github.com/Quantus-Network/quantus-miner), Apache-2.0.
