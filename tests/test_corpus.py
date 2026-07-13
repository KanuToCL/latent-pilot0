"""Corpus preflight + manifest tests (Mac, synth clips)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import soundfile as sf

from pilot0.audio.synth import synth_clip
from pilot0.corpus import (
    PreflightConfig,
    assign_splits,
    build_manifest,
    conditions,
    grid_signature,
    preflight,
    renderable_conditions,
    renderable_rows,
)
from pilot0.corpus.io import write_audio
from pilot0.corpus.loudness import measure_lufs

SR = 48000


@pytest.fixture
def corpus(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        write_audio(src / f"c{i}.wav", synth_clip(i, sr=SR), SR)
    return preflight(src, tmp_path / "out")


def _speaker_corpus(tmp_path, speakers=("A", "B"), per=3):
    """Corpus where several clips share a speaker group (spk<X>_<i>.wav)."""
    src = tmp_path / "src"
    src.mkdir()
    seed = 0
    for spk in speakers:
        for i in range(per):
            write_audio(src / f"spk{spk}_{i}.wav", synth_clip(seed, sr=SR), SR)
            seed += 1
    return preflight(src, tmp_path / "out", group_fn=lambda p: p.stem.split("_")[0])


# --- preflight ----------------------------------------------------------------


def test_preflight_accepts_all_valid(corpus):
    assert len(corpus.accepted) == 6
    assert corpus.rejected == ()


def test_masters_normalized_to_minus_23_lufs(corpus):
    for rec in corpus.accepted:
        wav, sr = sf.read(corpus.norm_dir / f"{rec.token}.wav", dtype="float64")
        assert measure_lufs(wav, sr) == pytest.approx(-23.0, abs=0.5)


def test_records_carry_group_arm_and_peak(corpus):
    rec = corpus.accepted[0]
    assert rec.group_id == "c0"  # default per-stem group
    assert rec.arm == "default"
    assert np.isfinite(rec.post_norm_dbtp)


def test_tokens_are_unique(corpus):
    tokens = [r.token for r in corpus.accepted]
    assert len(set(tokens)) == len(tokens)


def test_preflight_is_deterministic(tmp_path):
    def run(sub):
        src = tmp_path / sub / "src"
        src.mkdir(parents=True)
        for i in range(4):
            write_audio(src / f"c{i}.wav", synth_clip(i, sr=SR), SR)
        return preflight(src, tmp_path / sub / "out")

    assert [r.token for r in run("a").accepted] == [r.token for r in run("b").accepted]


def test_preflight_rejects_short_clipped_and_dc(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_audio(src / "good.wav", synth_clip(0, sr=SR), SR)
    write_audio(src / "short.wav", synth_clip(1, sr=SR, seconds=0.2), SR)
    write_audio(src / "clipped.wav", np.clip(4.0 * synth_clip(2, sr=SR), -1.0, 1.0), SR)  # symmetric hard-clip
    write_audio(src / "dc.wav", synth_clip(3, sr=SR) + 0.1, SR)  # +0.1 DC offset

    reasons = dict(preflight(src, tmp_path / "out").rejected)
    assert "too short" in reasons["short.wav"]
    assert reasons["clipped.wav"] == "already clipped"
    assert reasons["dc.wav"] == "dc offset"


def test_preflight_rejects_unexpected_sample_rate(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_audio(src / "wrong.wav", synth_clip(0, sr=16000), 16000)
    pf = preflight(src, tmp_path / "out", PreflightConfig(expected_sr=48000))
    assert "unexpected sample rate" in dict(pf.rejected)["wrong.wav"]


def test_preflight_dedupes_identical_content(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_audio(src / "a.wav", synth_clip(0, sr=SR), SR)
    write_audio(src / "b.wav", synth_clip(0, sr=SR), SR)
    pf = preflight(src, tmp_path / "out")
    assert len(pf.accepted) == 1
    assert dict(pf.rejected)["b.wav"] == "duplicate content (== a.wav)"


# --- manifest & splits --------------------------------------------------------


def test_conditions_are_clean_plus_grid():
    conds = conditions()
    assert conds[0] == ("clean", 0)
    assert len(conds) == 36  # 1 clean + 7 families × 5 severities


def test_manifest_row_count_is_sources_times_conditions(corpus):
    man = build_manifest(corpus)
    assert man["n_conditions"] == 36
    assert man["n_rows"] == len(corpus.accepted) * 36


def test_splits_are_speaker_disjoint_and_non_empty(tmp_path):
    man = build_manifest(_speaker_corpus(tmp_path))
    splits_per_group: dict[str, set[str]] = defaultdict(set)
    for row in man["rows"]:
        splits_per_group[row["group"]].add(row["split"])
    assert all(len(s) == 1 for s in splits_per_group.values())  # no speaker crosses
    assert {row["split"] for row in man["rows"]} == {"train", "test"}  # both present
    assert man["n_groups"] == 2


def test_assign_splits_is_deterministic(tmp_path):
    pf = _speaker_corpus(tmp_path)
    assert assign_splits(pf.accepted) == assign_splits(pf.accepted)


def test_assert_grouped_rejects_ungrouped_multiclip(tmp_path):
    build_manifest(_speaker_corpus(tmp_path), assert_grouped=True)  # 2 speaker groups → fine

    src = tmp_path / "raw"
    src.mkdir()
    for i in range(4):
        write_audio(src / f"spkA_{i}.wav", synth_clip(i, sr=SR), SR)
    ungrouped = preflight(src, tmp_path / "out3")  # default stem group → 4 groups for 4 clips
    with pytest.raises(ValueError, match="assert_grouped"):
        build_manifest(ungrouped, assert_grouped=True)


def test_renderable_conditions_gate_bandlimit_at_16k():
    r16 = set(renderable_conditions(16000))  # Nyquist 8000
    assert ("bandlimit", 1) not in r16 and ("bandlimit", 2) not in r16  # 12k, 8k
    assert ("bandlimit", 3) in r16 and ("clean", 0) in r16
    assert len(renderable_conditions(48000)) == 36


def test_renderable_rows_drops_invalid_cells(corpus):
    man = build_manifest(corpus)
    rows16 = renderable_rows(man, 16000)
    assert all(not (r["family"] == "bandlimit" and r["severity"] in (1, 2)) for r in rows16)
    assert len(renderable_rows(man, 48000)) == man["n_rows"]  # all valid at 48k


def test_manifest_pins_reproducibility_params(corpus):
    man = build_manifest(corpus, test_frac=0.2)
    assert man["target_lufs"] == -23.0
    assert man["test_frac"] == 0.2
    assert man["split_algo"]
    assert man["grid_signature"] == grid_signature()
    prov = man["provenance"]
    assert prov["numpy"] != "unknown" and prov["pyloudnorm"] != "unknown" and prov["pilot0"]
