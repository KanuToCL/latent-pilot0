"""Audio helpers. Phase 0 ships synthetic clips only; real corpora arrive in
Phase 2 (VCTK / LibriSpeech / MUSDB / FSD50K)."""

from .synth import synth_batch, synth_clip

__all__ = ["synth_batch", "synth_clip"]
