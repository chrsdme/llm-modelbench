"""Hardware detection and live telemetry.

Cross-platform by design. It probes, in order, nvidia-smi, then rocm-smi, then Apple's
metal stats, and if none are present it degrades gracefully: the benchmark still runs, it
just reports VRAM/power/temp as unavailable instead of crashing. This is what lets the tool
run on any machine rather than one specific GPU.

Telemetry samples in a background thread while a task runs and reports peak VRAM, mean/
sustained power, and peak temperature. Everything here is best-effort and never fatal.
"""
from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GPUInfo:
    vendor: str = "none"          # nvidia | amd | apple | none
    name: str = "unknown"
    total_vram_gb: float = 0.0
    can_read_power: bool = False
    can_read_temp: bool = False
    driver_version: str | None = None


def _run(cmd: List[str], timeout: int = 5) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def detect_gpu() -> GPUInfo:
    """Best-effort single-GPU detection. Returns a 'none' vendor if nothing is readable."""
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
        if out:
            line = out.strip().splitlines()[0]
            parts = [x.strip() for x in line.split(",")]
            name, mem = parts[:2]
            driver = parts[2] if len(parts) > 2 else None
            return GPUInfo("nvidia", name, round(float(mem) / 1024, 1), True, True, driver)
    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        vram = 0.0
        if out:
            for tok in out.replace(",", " ").split():
                if tok.isdigit() and int(tok) > 1_000_000:
                    vram = max(vram, round(int(tok) / 1e9, 1))
        return GPUInfo("amd", "AMD GPU", vram, bool(_run(["rocm-smi", "--showpower"])), True)
    # Apple Silicon: unified memory, no discrete VRAM figure; report system RAM as a hint
    if _run(["sysctl", "-n", "hw.memsize"]):
        mem = _run(["sysctl", "-n", "hw.memsize"])
        gb = round(int(mem.strip()) / 1e9, 1) if mem and mem.strip().isdigit() else 0.0
        return GPUInfo("apple", "Apple Silicon (unified)", gb, False, False)
    return GPUInfo()


def suggested_vram_budget_gb(gpu: GPUInfo, headroom_gb: float = 1.5) -> float:
    """Leave headroom for the OS/display. Falls back to a safe 12GB when unknown."""
    if gpu.total_vram_gb > 0:
        return max(2.0, round(gpu.total_vram_gb - headroom_gb, 1))
    return 12.0


class Telemetry:
    """Background sampler. No-op (returns zeros) when the GPU can't be read."""

    def __init__(self, gpu: GPUInfo, interval: float = 0.1):
        self.gpu = gpu
        self.interval = interval
        self._samples: List[Tuple[float, float, float]] = []   # vram_mb, power_w, temp_c
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _query(self) -> Optional[Tuple[float, float, float]]:
        if self.gpu.vendor == "nvidia":
            out = _run(["nvidia-smi",
                        "--query-gpu=memory.used,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits"], timeout=2)
            if out:
                try:
                    v, p, t = [float(x) for x in out.strip().splitlines()[0].split(",")]
                    return v, p, t
                except Exception:
                    return None
        return None

    def start(self) -> None:
        if self.gpu.vendor != "nvidia":
            return
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while self._running:
            s = self._query()
            if s:
                self._samples.append(s)
            time.sleep(self.interval)

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if not self._samples:
            return {"vram_peak_mb": None, "power_mean_w": None, "temp_peak_c": None}
        vram = [s[0] for s in self._samples]
        power = [s[1] for s in self._samples]
        temp = [s[2] for s in self._samples]
        # drop the first 10% (load spike) for a sustained power figure
        sus = power[max(1, len(power) // 10):] or power
        return {
            "vram_peak_mb": round(max(vram), 1),
            "power_mean_w": round(statistics.mean(sus), 1),
            "temp_peak_c": round(max(temp), 1),
        }

    def current_temp(self) -> Optional[float]:
        s = self._query()
        return s[2] if s else None


class ProbeTelemetry:
    """Per-probe resource sampler for long-context/model-card evidence.

    The sampler records GPU, system RAM/swap, CPU, and the aggregate memory
    footprint of local Ollama/llama-server processes.  All measurements are
    best-effort and additive.  They are evidence, not an excuse to fail a
    benchmark when a sensor is unavailable.

    ``ram_delta_peak_mb`` is system-wide and can include unrelated activity.
    ``ollama_pss_delta_peak_mb`` (or RSS when PSS is unreadable) is the preferred
    host-memory signal for offload-aware estimates.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = max(0.05, float(interval))
        self._samples: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started_at: Optional[float] = None
        self._baseline: dict = {}
        self._prev_cpu: Optional[Tuple[int, int]] = None

    def _query_raw(self) -> dict:
        snap = _read_proc_meminfo()
        snap.update(_read_ollama_process_memory())
        snap.update(nvidia_live())
        snap['cpu_temp_c'] = _cpu_temp_c()
        snap['_cpu_stat'] = _read_proc_stat()
        return snap

    def _query(self) -> dict:
        snap = self._query_raw()
        cur_cpu = snap.pop('_cpu_stat', None)
        snap['cpu_usage_pct'] = cpu_usage_pct(self._prev_cpu, cur_cpu)
        self._prev_cpu = cur_cpu
        return snap

    def start(self) -> None:
        self._samples = []
        self._prev_cpu = None
        self._baseline = self._query()
        self._started_at = time.perf_counter()
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while self._running:
            try:
                self._samples.append(self._query())
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    @staticmethod
    def _numeric(samples: List[dict], key: str) -> List[float]:
        return [float(s[key]) for s in samples if isinstance(s.get(key), (int, float))]

    def stop(self) -> dict:
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval * 4))
        try:
            self._samples.append(self._query())
        except Exception:
            pass
        elapsed = round(time.perf_counter() - self._started_at, 3) if self._started_at is not None else None
        samples = self._samples or ([self._baseline] if self._baseline else [])

        def peak(key: str) -> Optional[float]:
            vals = self._numeric(samples, key)
            return round(max(vals), 1) if vals else None

        def minimum(key: str) -> Optional[float]:
            vals = self._numeric(samples, key)
            return round(min(vals), 1) if vals else None

        def mean(key: str) -> Optional[float]:
            vals = self._numeric(samples, key)
            return round(statistics.mean(vals), 1) if vals else None

        def value(key: str) -> Optional[float]:
            raw = self._baseline.get(key)
            return round(float(raw), 1) if isinstance(raw, (int, float)) else None

        def delta(high: Optional[float], start_value: object) -> Optional[float]:
            if high is None or not isinstance(start_value, (int, float)):
                return None
            return round(max(0.0, float(high) - float(start_value)), 1)

        ram_peak = peak('ram_used_mb')
        swap_peak = peak('swap_used_mb')
        vram_peak = peak('vram_used_mb')
        ollama_rss_peak = peak('ollama_rss_mb')
        ollama_pss_peak = peak('ollama_pss_mb')
        ollama_swap_peak = peak('ollama_swap_mb')

        return {
            'elapsed_seconds': elapsed,
            'telemetry_samples': len(samples),
            'vram_start_mb': value('vram_used_mb'),
            'vram_peak_mb': vram_peak,
            'vram_delta_peak_mb': delta(vram_peak, self._baseline.get('vram_used_mb')),
            'gpu_util_mean_pct': mean('gpu_util_pct'),
            'gpu_util_peak_pct': peak('gpu_util_pct'),
            'power_mean_w': mean('gpu_power_w'),
            'power_peak_w': peak('gpu_power_w'),
            'temp_peak_c': peak('gpu_temp_c'),
            'cpu_util_mean_pct': mean('cpu_usage_pct'),
            'cpu_util_peak_pct': peak('cpu_usage_pct'),
            'cpu_temp_peak_c': peak('cpu_temp_c'),
            'ram_start_mb': value('ram_used_mb'),
            'ram_peak_mb': ram_peak,
            'ram_delta_peak_mb': delta(ram_peak, self._baseline.get('ram_used_mb')),
            'ram_available_start_mb': value('ram_available_mb'),
            'ram_available_min_mb': minimum('ram_available_mb'),
            'ram_total_mb': self._baseline.get('ram_total_mb'),
            'swap_start_mb': value('swap_used_mb'),
            'swap_peak_mb': swap_peak,
            'swap_delta_peak_mb': delta(swap_peak, self._baseline.get('swap_used_mb')),
            'swap_total_mb': self._baseline.get('swap_total_mb'),
            'ollama_process_count_start': self._baseline.get('ollama_process_count'),
            'ollama_process_count_peak': peak('ollama_process_count'),
            'ollama_rss_start_mb': value('ollama_rss_mb'),
            'ollama_rss_peak_mb': ollama_rss_peak,
            'ollama_rss_delta_peak_mb': delta(ollama_rss_peak, self._baseline.get('ollama_rss_mb')),
            'ollama_pss_start_mb': value('ollama_pss_mb'),
            'ollama_pss_peak_mb': ollama_pss_peak,
            'ollama_pss_delta_peak_mb': delta(ollama_pss_peak, self._baseline.get('ollama_pss_mb')),
            'ollama_swap_start_mb': value('ollama_swap_mb'),
            'ollama_swap_peak_mb': ollama_swap_peak,
            'ollama_swap_delta_peak_mb': delta(ollama_swap_peak, self._baseline.get('ollama_swap_mb')),
            'host_memory_signal': (
                'ollama_pss_delta' if ollama_pss_peak is not None
                else 'ollama_rss_delta' if ollama_rss_peak is not None
                else 'system_ram_delta' if ram_peak is not None
                else 'unavailable'
            ),
        }

# ---- live watcher telemetry -------------------------------------------------

def _read_proc_meminfo() -> dict:
    vals = {}
    try:
        for line in open('/proc/meminfo', 'r', encoding='utf-8'):
            parts = line.split()
            if len(parts) >= 2:
                vals[parts[0].rstrip(':')] = float(parts[1]) / 1024.0  # MB
    except Exception:
        return {}
    total = vals.get('MemTotal')
    available = vals.get('MemAvailable')
    swap_total = vals.get('SwapTotal')
    swap_free = vals.get('SwapFree')
    return {
        'ram_total_mb': round(total, 1) if total is not None else None,
        'ram_used_mb': round(total - available, 1) if total is not None and available is not None else None,
        'ram_available_mb': round(available, 1) if available is not None else None,
        'ram_used_pct': round((total - available) / total * 100.0, 1) if total and available is not None else None,
        'swap_total_mb': round(swap_total, 1) if swap_total is not None else None,
        'swap_used_mb': round(swap_total - swap_free, 1) if swap_total is not None and swap_free is not None else None,
    }


def _read_ollama_process_memory() -> dict:
    """Best-effort aggregate host-memory footprint for local Ollama runners.

    Only aggregate counters are returned.  Command lines and environments are
    never persisted.  PSS is preferred because shared pages are not double
    counted; RSS is retained as a fallback for systems that restrict
    ``smaps_rollup``.
    """
    rss_kb = 0.0
    pss_kb = 0.0
    swap_kb = 0.0
    pss_readable = False
    count = 0
    try:
        entries = list(os.scandir('/proc'))
    except Exception:
        return {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        base = f'/proc/{entry.name}'
        try:
            comm = open(f'{base}/comm', 'r', encoding='utf-8', errors='replace').read().strip()
        except Exception:
            comm = ''
        try:
            cmdline = open(f'{base}/cmdline', 'rb').read(4096).replace(b'\0', b' ').decode('utf-8', 'replace')
        except Exception:
            cmdline = ''
        identity = f'{comm} {cmdline}'.lower()
        if not any(token in identity for token in ('ollama', 'llama_server', 'llama-server')):
            continue
        count += 1
        status_vals = {}
        try:
            for line in open(f'{base}/status', 'r', encoding='utf-8', errors='replace'):
                if line.startswith(('VmRSS:', 'VmSwap:')):
                    parts = line.split()
                    if len(parts) >= 2:
                        status_vals[parts[0].rstrip(':')] = float(parts[1])
        except Exception:
            pass
        rss_kb += status_vals.get('VmRSS', 0.0)
        swap_kb += status_vals.get('VmSwap', 0.0)
        try:
            for line in open(f'{base}/smaps_rollup', 'r', encoding='utf-8', errors='replace'):
                if line.startswith('Pss:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        pss_kb += float(parts[1])
                        pss_readable = True
                    break
        except Exception:
            pass
    if not count:
        return {'ollama_process_count': 0, 'ollama_rss_mb': 0.0, 'ollama_pss_mb': None, 'ollama_swap_mb': 0.0}
    return {
        'ollama_process_count': count,
        'ollama_rss_mb': round(rss_kb / 1024.0, 1),
        'ollama_pss_mb': round(pss_kb / 1024.0, 1) if pss_readable else None,
        'ollama_swap_mb': round(swap_kb / 1024.0, 1),
    }


def _read_proc_stat() -> Optional[Tuple[int, int]]:
    try:
        line = open('/proc/stat', 'r', encoding='utf-8').readline()
        nums = [int(x) for x in line.split()[1:]]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        return idle, total
    except Exception:
        return None


def cpu_usage_pct(prev: Optional[Tuple[int, int]], cur: Optional[Tuple[int, int]]) -> Optional[float]:
    if not prev or not cur:
        return None
    idle_delta = cur[0] - prev[0]
    total_delta = cur[1] - prev[1]
    if total_delta <= 0:
        return None
    return round((1.0 - idle_delta / total_delta) * 100.0, 1)


def _cpu_temp_c() -> Optional[float]:
    # Best effort Linux hwmon/thermal. Never fatal.
    paths = []
    import glob
    paths.extend(glob.glob('/sys/class/thermal/thermal_zone*/temp'))
    paths.extend(glob.glob('/sys/class/hwmon/hwmon*/temp*_input'))
    vals = []
    for path in paths:
        try:
            v = float(open(path, 'r', encoding='utf-8').read().strip())
            if v > 1000:
                v = v / 1000.0
            if 10 <= v <= 120:
                vals.append(v)
        except Exception:
            pass
    return round(max(vals), 1) if vals else None


def nvidia_live() -> dict:
    """Single-GPU live snapshot for the watcher."""
    if not shutil.which('nvidia-smi'):
        return {}
    out = _run(['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'], timeout=2)
    if not out:
        return {}
    try:
        name, temp, util, mem_used, mem_total, power = [x.strip() for x in out.strip().splitlines()[0].split(',')[:6]]
        mem_used_f = float(mem_used); mem_total_f = float(mem_total)
        return {
            'gpu_name': name,
            'gpu_temp_c': round(float(temp), 1),
            'gpu_util_pct': round(float(util), 1),
            'vram_used_mb': round(mem_used_f, 1),
            'vram_total_mb': round(mem_total_f, 1),
            'vram_used_pct': round(mem_used_f / mem_total_f * 100.0, 1) if mem_total_f else None,
            'gpu_power_w': round(float(power), 1),
        }
    except Exception:
        return {}


def live_snapshot(prev_cpu: Optional[Tuple[int, int]] = None) -> Tuple[dict, Optional[Tuple[int, int]]]:
    cur_cpu = _read_proc_stat()
    snap = {}
    snap.update(_read_proc_meminfo())
    snap.update(nvidia_live())
    snap['cpu_usage_pct'] = cpu_usage_pct(prev_cpu, cur_cpu)
    snap['cpu_temp_c'] = _cpu_temp_c()
    return snap, cur_cpu


def host_memory_snapshot() -> dict:
    """Public best-effort host RAM/swap snapshot for pre-probe safety gates."""
    return _read_proc_meminfo()
