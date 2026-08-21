"""Live operator panel for Job B encode runs. Operator tooling only — reads the
cache directory and the job log; never touches the pipeline.

Usage (own terminal window):  .venv/Scripts/python tools/encode_monitor.py
"""
from __future__ import annotations

import json
import subprocess
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


def cached_counts() -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    if not CACHE.exists():
        return out
    for version_dir in CACHE.iterdir():
        for name_dir in (d for d in version_dir.iterdir() if d.is_dir()):
            for var_dir in (d for d in name_dir.iterdir() if d.is_dir()):
                out[(name_dir.name, var_dir.name)] = sum(1 for _ in var_dir.glob("*.npz"))
    return out


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


def render(expected, history, t0) -> Panel:
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
        (f"   ETA {eta / 3600:.1f} h" if eta != float("inf") else "   ETA —", "green"),
        (f"   elapsed {(now - t0) / 60:.0f} min", "dim"),
    )
    tail = Text("\n".join(log_tail()), style="dim")
    status = Text("ENCODE COMPLETE — gate 1 phase (watch log)", style="bold black on bright_green") \
        if done >= total else Text("")
    return Panel(Group(hdr, bar, Text(""), table, Text(""), res, Text(""), tail, status),
                 title="[bold bright_green]pilot0 // job B — real mini-sweep[/]",
                 border_style="green")


def main() -> None:
    console = Console()
    expected = expected_counts()
    history: deque = deque(maxlen=60)  # ~2 min rate window
    t0 = time.time()
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            live.update(render(expected, history, t0))
            time.sleep(REFRESH_S)


if __name__ == "__main__":
    main()
