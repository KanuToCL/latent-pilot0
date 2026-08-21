"""Live operator panel + controller for Job B encode runs. Reads the cache
directory and the job log; owns the job subprocess (Ctrl+O start / Ctrl+P pause).
Pausing is always safe: the cache is atomic + resume-safe.

Usage (own terminal window):  .venv/Scripts/python tools/encode_monitor.py [--autostart]
The job runs DETACHED: closing this window leaves it running; reopening the
panel reattaches via data/job_b.pid. Only ^P (or taskkill) stops the job.
"""
from __future__ import annotations

import json
import msvcrt
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from pilot0.corpus.manifest import renderable_conditions
from pilot0.seam.registry import REAL_SPECS, make_encoder

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "speech"
MANIFEST = ROOT / "data" / "corpus" / "speech" / "manifest.json"
LOG = ROOT / "data" / "job_b.log"
REFRESH_S = 2.0

ENCODERS: dict[str, list[str]] = {
    "logmel": ["mel"],
    "energy": ["energy"],
    **{n: list(s["variants"]) for n, s in REAL_SPECS.items()},
}


def expected_counts() -> dict[tuple[str, str], int]:
    n_sources = json.loads(MANIFEST.read_text())["n_sources"]
    out = {}
    for name, variants in ENCODERS.items():
        n_cells = n_sources * len(renderable_conditions(make_encoder(name).native_sr))
        for v in variants:
            out[(name, v)] = n_cells
    return out


_count_cache: tuple[float, dict] = (0.0, {})
COUNT_TTL_S = 20.0  # audit: don't re-walk 338k files every 2 s against the encoder


def cached_counts() -> dict[tuple[str, str], int]:
    global _count_cache
    ts, cached = _count_cache
    if time.time() - ts < COUNT_TTL_S:
        return cached
    out: dict[tuple[str, str], int] = {}
    if CACHE.exists():
        for version_dir in CACHE.iterdir():
            for name_dir in (d for d in version_dir.iterdir() if d.is_dir() and d.name != "headroom"):
                for var_dir in (d for d in name_dir.iterdir() if d.is_dir()):
                    key = (name_dir.name, var_dir.name)  # sum across cache versions (audit note)
                    out[key] = out.get(key, 0) + sum(1 for _ in var_dir.glob("*.npz"))
    _count_cache = (time.time(), out)
    return out


PIDFILE = ROOT / "data" / "job_b.pid"


class JobController:
    """Controls a DETACHED job_b_run.py: the encoder survives this panel closing
    (post-audit — the old parent/child design killed overnight runs on window
    close). Reattaches via pid file. Kill is safe mid-encode: atomic cache."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._log = None

    def _pidfile_pid(self) -> int | None:
        try:
            pid = int(PIDFILE.read_text())
            if psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
                return pid
        except (FileNotFoundError, ValueError, psutil.Error):
            pass
        return None

    @property
    def pid(self) -> int | None:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc.pid
        return self._pidfile_pid()

    @property
    def running(self) -> bool:
        return self.pid is not None

    def start(self) -> None:
        if self.running:
            return
        env = {**os.environ,
               "HF_HOME": str(ROOT / "data" / "hf-cache"),
               "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
               "PYTHONIOENCODING": "utf-8"}
        LOG.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(LOG, "ab")
        self.proc = subprocess.Popen(
            [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-u", str(ROOT / "tools" / "job_b_run.py")],
            stdout=self._log, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
        PIDFILE.write_text(str(self.proc.pid))

    def stop(self) -> None:
        pid = self.pid
        if pid is not None:  # /T kills grandchildren (ffmpeg), /F is safe: atomic cache
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        if self._log:
            self._log.close()
            self._log = None
        PIDFILE.unlink(missing_ok=True)


def job_status(job: JobController) -> Text:
    pid = job.pid
    if pid is not None:
        return Text.assemble(("● RUNNING (detached — survives closing this window)", "bold bright_green"),
                             (f"  pid {pid}", "dim"), ("    ^P pause   ^O start", "dim"))
    return Text.assemble(("■ NOT RUNNING — cache kept, ^O resumes where it left off", "bold yellow"),
                         ("    ^O start   ^P pause", "dim"))


def headroom_status() -> Text:
    """The headroom scan renders every cell at a rate BEFORE caching anything —
    without this line the panel looks dead during that (one-time, persisted) phase."""
    rates = {make_encoder(n).native_sr for n in ENCODERS}
    done = {int(p.stem.split("_")[1]) for p in CACHE.glob("*/headroom/sr_*.json")} if CACHE.exists() else set()
    if rates <= done:
        return Text("headroom scalars: all rates ready", style="green")
    pend = ", ".join(f"{r // 1000}k" for r in sorted(rates - done))
    if sum(cached_counts().values()) > 0:  # latents flowing; remaining scans run when their encoder starts
        return Text(f"headroom: pending {pend} (each scan runs when its encoder starts)", style="dim")
    return Text(f"phase: headroom scan (one-time per rate; pending: {pend}) — no latents cached until the first scan completes",
                style="bold yellow")


def gpu_stats() -> str:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(", ")
        return f"GPU {q[0]:>3s}%  VRAM {int(q[1]) / 1024:.1f}/{int(q[2]) / 1024:.1f} GB  {q[3]}°C  {float(q[4]):.0f} W"
    except Exception:
        return "GPU: n/a"


def log_tail(n: int = 4) -> list[str]:
    if not LOG.exists():
        return ["(no job log yet)"]
    lines = LOG.read_text(errors="replace").splitlines()
    return [ln.strip()[:110] for ln in lines[-n:] if ln.strip()] or ["(log empty)"]


def render(expected, history, t0, job: JobController) -> Panel:
    counts = cached_counts()
    done = sum(counts.values())
    total = sum(expected.values())
    now = time.time()
    history.append((now, done))
    rate = 0.0
    if len(history) > 1:
        (ta, ca), (tb, cb) = history[0], history[-1]
        rate = (cb - ca) / max(tb - ta, 1e-9)  # files/s over window
    eta = (total - done) / rate if rate > 0 else float("inf")

    table = Table(box=None, pad_edge=False, header_style="bold bright_green")
    table.add_column("encoder", style="cyan", width=12)
    table.add_column("variants done / expected", width=34)
    table.add_column("%", justify="right", width=6)
    for name, variants in ENCODERS.items():
        d = sum(counts.get((name, v), 0) for v in variants)
        e = sum(expected[(name, v)] for v in variants)
        pct = 100 * d / e if e else 0
        mark = "bright_green" if d >= e else ("yellow" if d else "dim")
        table.add_row(name, f"[{mark}]{d:6d} / {e:6d}[/]", f"{pct:5.1f}")

    bar = ProgressBar(total=total, completed=done, width=58, complete_style="bright_green")
    vm = psutil.virtual_memory()
    res = Text.assemble(
        (gpu_stats(), "bright_cyan"), "   ",
        (f"CPU {psutil.cpu_percent():4.1f}%  RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.0f} GB", "bright_magenta"),
    )
    hdr = Text.assemble(
        (f"{done:,} / {total:,} latents", "bold bright_green"),
        (f"   {rate * 60:,.0f}/min", "green"),
        (f"   ETA~ {eta / 3600:.1f} h (rough: per-encoder cost varies ~100x)" if eta != float("inf") else "   ETA —", "green"),
        (f"   elapsed {(now - t0) / 60:.0f} min", "dim"),
    )
    tail = Text("\n".join(log_tail()), style="dim")
    status = Text("ENCODE COMPLETE — gate 1 phase (watch log)", style="bold black on bright_green") \
        if done >= total else Text("")
    return Panel(Group(job_status(job), Text(""), hdr, bar, Text(""), table, Text(""), res, headroom_status(), Text(""), tail, status),
                 title="[bold bright_green]pilot0 // job B — real mini-sweep[/]",
                 border_style="green")


def main() -> None:
    console = Console()
    expected = expected_counts()
    history: deque = deque(maxlen=60)  # ~2 min rate window
    t0 = time.time()
    job = JobController()
    if "--autostart" in sys.argv:
        job.start()
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                live.update(render(expected, history, t0, job))
                # poll keys in small slices so ^O/^P feel instant
                deadline = time.time() + REFRESH_S
                while time.time() < deadline:
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b"\x0f":   # Ctrl+O
                            job.start()
                        elif key == b"\x10":  # Ctrl+P
                            job.stop()
                    time.sleep(0.05)
    except KeyboardInterrupt:
        pass  # panel exit NEVER kills the detached job (audit: overnight-run safety)


if __name__ == "__main__":
    main()
