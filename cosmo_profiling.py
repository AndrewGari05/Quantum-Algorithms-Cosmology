# =============================================================================
#  cosmo_profiling.py — Resource profiling (CPU/GPU memory, wall time, GPU-hours)
# =============================================================================
#
#  PURPOSE.  To plan and justify the compute footprint of the pipeline on an
#  HPC cluster (Nicte-Ha at IBERO, and the new RTX PRO 6000 / NVIDIA cluster),
#  every run can now be PROFILED: peak host (RAM) memory, peak device (VRAM)
#  memory when a GPU is used, wall-clock time, and the derived GPU-hours.
#
#  WHAT IT PRODUCES.
#    * A background sampler thread records (t, RSS_MB, VRAM_MB, host_%, gpu_%)
#      at a fixed cadence with negligible overhead.
#    * A summary dict: peak RSS, peak VRAM, wall time, GPU-hours, mean GPU util.
#    * A figure `resource_usage_<tag>.png` with two stacked panels:
#        - memory (host RSS and, if present, device VRAM) vs time,
#        - utilization (host CPU% and GPU%) vs time,
#      annotated with the peaks and the GPU-hours, saved next to the run's
#      other outputs (works headless — uses whatever backend is active).
#
#  DEPENDENCIES.  `psutil` for host memory/CPU (already a common dependency).
#  GPU metrics use NVIDIA's NVML through `pynvml` if installed, else a parse of
#  `nvidia-smi`; if neither is available the GPU panel is simply omitted, so the
#  profiler NEVER breaks a CPU-only run (e.g. on Nicte-Ha without GPUs).
#
#  This module is import-safe everywhere and has no project dependencies, so it
#  can be reused by cosmo_modular_quantum.py and cosmo_genetic_optimizers.py.
# =============================================================================

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import psutil
    _PSUTIL_OK = True
except Exception:                                       # pragma: no cover
    _PSUTIL_OK = False


# =============================================================================
# 1.  GPU DETECTION & MEMORY QUERY  (NVML preferred, nvidia-smi fallback)
# =============================================================================

class _GPUMonitor:
    """Query NVIDIA GPU memory and utilization, if a GPU is visible.

    Tries NVML (pynvml) first — fast, in-process, no subprocess. Falls back to
    parsing `nvidia-smi`. If neither works, `available` is False and all
    queries return zeros, so the rest of the profiler degrades gracefully on a
    CPU-only node.
    """

    def __init__(self):
        self.available = False
        self._backend = None
        self._handle = None
        self._pynvml = None
        self._init_nvml() or self._init_smi()

    def _init_nvml(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() < 1:
                return False
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._backend = 'nvml'
            self.available = True
            return True
        except Exception:
            return False

    def _init_smi(self) -> bool:
        if shutil.which('nvidia-smi') is None:
            return False
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                self._backend = 'smi'
                self.available = True
                return True
        except Exception:
            pass
        return False

    def name(self) -> str:
        """Human-readable GPU model (or 'none')."""
        if not self.available:
            return 'none'
        try:
            if self._backend == 'nvml':
                n = self._pynvml.nvmlDeviceGetName(self._handle)
                return n.decode() if isinstance(n, bytes) else str(n)
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=name',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5)
            return out.stdout.strip().splitlines()[0]
        except Exception:
            return 'unknown NVIDIA GPU'

    def total_mem_mb(self) -> float:
        """Total VRAM in MB (0 if no GPU)."""
        if not self.available:
            return 0.0
        try:
            if self._backend == 'nvml':
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return info.total / 1024.0 / 1024.0
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return 0.0

    def sample(self):
        """Return (vram_used_MB, gpu_util_percent). Zeros if no GPU."""
        if not self.available:
            return 0.0, 0.0
        try:
            if self._backend == 'nvml':
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                return mem.used / 1024.0 / 1024.0, float(util.gpu)
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            used, util = out.stdout.strip().splitlines()[0].split(',')
            return float(used), float(util)
        except Exception:
            return 0.0, 0.0

    def close(self):
        if self._backend == 'nvml' and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


# =============================================================================
# 2.  THE PROFILER  (background sampling thread + summary + figure)
# =============================================================================

@dataclass
class ProfileResult:
    """Summary of one profiled run.

    Attributes:
        tag: short identifier (used in the figure filename).
        wall_s: wall-clock seconds the profiled block took.
        peak_rss_mb: peak host resident memory (MB).
        peak_vram_mb: peak device memory (MB); 0 if CPU-only.
        gpu_hours: device-hours = (wall_s/3600) when a GPU was active, else 0.
        mean_gpu_util: mean GPU utilization % over the run (0 if CPU-only).
        device: 'GPU' or 'CPU' (what the simulator actually used).
        gpu_name: GPU model string (or 'none').
        n_samples: number of monitor samples taken.
    """
    tag: str
    wall_s: float
    peak_rss_mb: float
    peak_vram_mb: float
    gpu_hours: float
    mean_gpu_util: float
    device: str
    gpu_name: str
    n_samples: int
    # raw traces (for plotting)
    t: List[float] = field(default_factory=list)
    rss: List[float] = field(default_factory=list)
    vram: List[float] = field(default_factory=list)
    cpu_util: List[float] = field(default_factory=list)
    gpu_util: List[float] = field(default_factory=list)

    def as_row(self) -> dict:
        """Compact dict for logging or CSV provenance."""
        return {
            'wall_s': round(self.wall_s, 2),
            'peak_rss_mb': round(self.peak_rss_mb, 1),
            'peak_vram_mb': round(self.peak_vram_mb, 1),
            'gpu_hours': round(self.gpu_hours, 5),
            'mean_gpu_util_pct': round(self.mean_gpu_util, 1),
            'device': self.device, 'gpu_name': self.gpu_name,
        }


class ResourceProfiler:
    """Sample host/device memory and utilization in a background thread.

    Usage:
        prof = ResourceProfiler(tag='lcdm_gpu', device='GPU', interval=0.25)
        prof.start()
        ...  # the work to profile
        result = prof.stop()
        prof.plot(result, outdir)

    The sampling thread reads psutil + NVML at `interval` seconds. Overhead is
    a few hundred microseconds per sample, negligible next to MCMC/VI/GA work.

    Args:
        tag: identifier for filenames.
        device: 'GPU' or 'CPU' — what the simulator is configured to use. Only
            affects how gpu_hours is attributed (GPU-hours accrue only when a
            GPU is the active device).
        interval: sampling cadence in seconds.
    """

    def __init__(self, tag: str = 'run', device: str = 'CPU',
                 interval: float = 0.25):
        self.tag = tag
        self.device = device
        self.interval = float(interval)
        self._gpu = _GPUMonitor()
        self._proc = psutil.Process(os.getpid()) if _PSUTIL_OK else None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._t0 = None
        self._t, self._rss, self._vram = [], [], []
        self._cpu, self._gutil = [], []

    def _loop(self):
        # Prime cpu_percent (first call returns 0.0 by design).
        if self._proc is not None:
            self._proc.cpu_percent(None)
        while not self._stop.is_set():
            now = time.time() - self._t0
            rss = (self._proc.memory_info().rss / 1024.0 / 1024.0
                   if self._proc is not None else 0.0)
            cpu = (self._proc.cpu_percent(None)
                   if self._proc is not None else 0.0)
            vram, gutil = self._gpu.sample()
            self._t.append(now); self._rss.append(rss); self._vram.append(vram)
            self._cpu.append(cpu); self._gutil.append(gutil)
            self._stop.wait(self.interval)

    def start(self):
        """Begin background sampling."""
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> ProfileResult:
        """Stop sampling and return a `ProfileResult` summary."""
        wall = time.time() - self._t0 if self._t0 else 0.0
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        gpu_active = self.device == 'GPU' and self._gpu.available
        peak_rss = max(self._rss) if self._rss else 0.0
        peak_vram = max(self._vram) if self._vram else 0.0
        mean_gutil = float(np.mean(self._gutil)) if self._gutil else 0.0
        gpu_hours = (wall / 3600.0) if gpu_active else 0.0
        res = ProfileResult(
            tag=self.tag, wall_s=wall, peak_rss_mb=peak_rss,
            peak_vram_mb=peak_vram, gpu_hours=gpu_hours,
            mean_gpu_util=mean_gutil,
            device='GPU' if gpu_active else 'CPU',
            gpu_name=self._gpu.name(), n_samples=len(self._t),
            t=list(self._t), rss=list(self._rss), vram=list(self._vram),
            cpu_util=list(self._cpu), gpu_util=list(self._gutil))
        self._gpu.close()
        return res

    # ── figure ───────────────────────────────────────────────────────────────
    @staticmethod
    def plot(result: ProfileResult, outdir: str,
             title_extra: str = '') -> Optional[str]:
        """Save a two-panel memory/utilization-vs-time figure.

        Uses the Matplotlib backend already active (Agg on HPC), so it is safe
        headless. Returns the PNG path, or None if there were no samples.
        """
        if result.n_samples == 0:
            return None
        import matplotlib.pyplot as plt
        os.makedirs(outdir, exist_ok=True)
        has_gpu = result.device == 'GPU' and any(v > 0 for v in result.vram)

        nrows = 2 if has_gpu else 2  # always 2 panels (mem, util)
        fig, (ax_mem, ax_util) = plt.subplots(2, 1, figsize=(10, 7),
                                              sharex=True)

        # memory panel
        ax_mem.plot(result.t, result.rss, '-', color='#1f77b4', lw=1.8,
                    label=f'Host RSS (peak {result.peak_rss_mb:.0f} MB)')
        ax_mem.axhline(result.peak_rss_mb, color='#1f77b4', ls=':', alpha=0.5)
        if has_gpu:
            ax_mem.plot(result.t, result.vram, '-', color='#2ca02c', lw=1.8,
                        label=f'Device VRAM (peak {result.peak_vram_mb:.0f} MB)')
            ax_mem.axhline(result.peak_vram_mb, color='#2ca02c', ls=':',
                           alpha=0.5)
        ax_mem.set_ylabel('Memory [MB]', fontsize=11)
        ax_mem.grid(True, alpha=0.3); ax_mem.legend(fontsize=9, loc='best')
        ax_mem.set_title('Memory usage vs time', fontsize=11)

        # utilization panel
        ax_util.plot(result.t, result.cpu_util, '-', color='#ff7f0e', lw=1.5,
                     label='Host CPU %')
        if has_gpu:
            ax_util.plot(result.t, result.gpu_util, '-', color='#d62728',
                         lw=1.5,
                         label=f'GPU % (mean {result.mean_gpu_util:.0f}%)')
        ax_util.set_ylabel('Utilization [%]', fontsize=11)
        ax_util.set_xlabel('Wall-clock time [s]', fontsize=11)
        ax_util.grid(True, alpha=0.3); ax_util.legend(fontsize=9, loc='best')
        ax_util.set_title('Compute utilization vs time', fontsize=11)

        dev = (f"{result.device} ({result.gpu_name})" if result.device == 'GPU'
               else "CPU")
        sup = (f"Resource profile — {result.tag}   |   device: {dev}   |   "
               f"wall {result.wall_s:.1f}s")
        if result.device == 'GPU':
            sup += f"   |   {result.gpu_hours:.4f} GPU-h"
        if title_extra:
            sup += f"\n{title_extra}"
        fig.suptitle(sup, fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(outdir, f'resource_usage_{result.tag}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        return path


def summarize(result: ProfileResult) -> str:
    """One-line human summary for logs/console."""
    s = (f"[profile] {result.tag}: wall={result.wall_s:.1f}s  "
         f"peak_RSS={result.peak_rss_mb:.0f}MB  device={result.device}")
    if result.device == 'GPU':
        s += (f"  peak_VRAM={result.peak_vram_mb:.0f}MB  "
              f"GPU-h={result.gpu_hours:.4f}  "
              f"meanGPU={result.mean_gpu_util:.0f}%")
    return s
