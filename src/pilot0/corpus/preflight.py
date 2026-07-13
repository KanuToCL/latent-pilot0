"""Preflight: validate source clips, loudness-normalise to −23 LUFS, and rename
to opaque content tokens. A clip that fails validation cannot enter the study
(fail-closed). Runs offline on the Mac; the real corpora drop into `src_dir`
later.

Each accepted source carries a `group_id` (speaker / track) and an `arm`
(speech48 / music / …), derived from the path by caller-supplied functions.
Splitting is done on the GROUP, not the clip, so the same speaker never crosses
train/test (§2.5) — the defaults treat each file as its own group, which is only
correct for one-clip-per-source corpora."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..degrade.base import true_peak_dbtp
from .io import read_audio, to_mono, write_audio
from .loudness import TARGET_LUFS, normalize_lufs
from .provenance import provenance
from .tokens import content_sha256, content_token


def _stem_group(path: Path) -> str:
    return path.stem


def _single_arm(path: Path) -> str:
    return "default"


@dataclass(frozen=True)
class PreflightConfig:
    min_seconds: float = 1.0
    max_seconds: float = 30.0
    max_clip_fraction: float = 1e-3
    max_dc: float = 0.02
    max_post_norm_dbtp: float = 12.0  # reject sources over-gained by normalisation
    expected_sr: int | None = None  # if set, reject any other rate (arm allowlist)
    target_lufs: float = TARGET_LUFS


@dataclass(frozen=True)
class SourceRecord:
    token: str
    group_id: str
    arm: str
    original: str
    sr: int
    n_samples: int
    duration_s: float
    orig_channels: int
    orig_lufs: float
    gain_db: float
    post_norm_dbtp: float
    content_sha: str


@dataclass(frozen=True)
class PreflightResult:
    accepted: tuple[SourceRecord, ...]
    rejected: tuple[tuple[str, str], ...]
    norm_dir: Path
    config: PreflightConfig


def _reject_reason(mono: np.ndarray, sr: int, dur: float, cfg: PreflightConfig) -> str | None:
    if not np.isfinite(mono).all():
        return "non-finite samples"
    if cfg.expected_sr is not None and sr != cfg.expected_sr:
        return f"unexpected sample rate ({sr} != {cfg.expected_sr})"
    if dur < cfg.min_seconds:
        return f"too short ({dur:.2f}s < {cfg.min_seconds}s)"
    if dur > cfg.max_seconds:
        return f"too long ({dur:.2f}s > {cfg.max_seconds}s)"
    if abs(float(np.mean(mono))) > cfg.max_dc:
        return "dc offset"
    if float(np.mean(np.abs(mono) >= 0.999)) > cfg.max_clip_fraction:
        return "already clipped"
    return None


def preflight(
    src_dir,
    out_dir,
    config: PreflightConfig = PreflightConfig(),
    *,
    group_fn: Callable[[Path], str] = _stem_group,
    arm_fn: Callable[[Path], str] = _single_arm,
) -> PreflightResult:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    norm_dir = out_dir / "norm"
    norm_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[SourceRecord] = []
    rejected: list[tuple[str, str]] = []
    first_by_token: dict[str, str] = {}

    for path in sorted(src_dir.glob("*.wav")):
        wav, sr = read_audio(path)
        channels = wav.shape[1] if wav.ndim > 1 else 1
        mono = to_mono(wav).astype(np.float64)
        dur = len(mono) / sr

        reason = _reject_reason(mono, sr, dur, config)
        if reason:
            rejected.append((path.name, reason))
            continue

        normalised, lufs, gain = normalize_lufs(mono, sr, config.target_lufs)
        if normalised is None:
            rejected.append((path.name, "loudness undefined"))
            continue

        peak_dbtp = true_peak_dbtp(normalised, sr)
        if peak_dbtp > config.max_post_norm_dbtp:
            rejected.append((path.name, f"over-gained after normalise ({peak_dbtp:.1f} dBTP)"))
            continue

        token = content_token(mono, sr)
        if token in first_by_token:
            rejected.append((path.name, f"duplicate content (== {first_by_token[token]})"))
            continue
        first_by_token[token] = path.name

        write_audio(norm_dir / f"{token}.wav", normalised, sr)
        accepted.append(
            SourceRecord(
                token=token,
                group_id=group_fn(path),
                arm=arm_fn(path),
                original=path.name,
                sr=sr,
                n_samples=len(mono),
                duration_s=dur,
                orig_channels=channels,
                orig_lufs=lufs,
                gain_db=float(20.0 * np.log10(gain)),
                post_norm_dbtp=float(peak_dbtp),
                content_sha=content_sha256(mono, sr),
            )
        )

    _write_report(out_dir, accepted, rejected, config)
    return PreflightResult(tuple(accepted), tuple(rejected), norm_dir, config)


def _write_report(out_dir: Path, accepted, rejected, config: PreflightConfig) -> None:
    doc = {
        "target_lufs": config.target_lufs,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "accepted": [asdict(a) for a in accepted],
        "rejected": [{"file": f, "reason": r} for f, r in rejected],
        "provenance": provenance(),
    }
    (out_dir / "preflight.json").write_text(json.dumps(doc, indent=2))
