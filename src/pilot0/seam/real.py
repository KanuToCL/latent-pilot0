"""RealCodecEncoder — lazy-loading adapters for the frozen backends (GPU box).

torch / transformers / dac are NOT installed on the Mac dev machine, so this
module is written but cannot be executed here. Constructing a real encoder on a
torch-less machine raises `RealBackendUnavailable`; downstream code falls back to
a `fake-*` encoder.

The encode paths target the models' documented APIs and are marked BRINGUP:
validated on the GPU box via docs/GPU_BRINGUP.md (shape + fake/real parity), NOT
on the Mac. Currently wired: EnCodec (continuous pre-quant `z`) and WavLM (per
layer, with the model's own feature-extractor normalization). RVQ-depth variants,
DAC and Mimi are wired at bring-up / Phase 5. The `LatentResult` contract is
unchanged, so the pipeline is drop-in the moment the box is reassembled.
"""

from __future__ import annotations

import numpy as np

from .base import LatentResult, measure_frame_rate

_LAYER_MAP = {"l1": 1, "l6": 6, "l12": 12, "l18": 18, "l24": 24}


class RealBackendUnavailable(RuntimeError):
    """Raised when a real backend is requested on a machine without torch."""


class RealCodecEncoder:
    def __init__(
        self,
        name: str,
        family: str,
        native_sr: int,
        latent_dim: int,
        checkpoint: str,
        variants: tuple[str, ...],
    ) -> None:
        self.name = name
        self.family = family
        self.native_sr = int(native_sr)
        self.latent_dim = int(latent_dim)
        self.checkpoint = checkpoint
        self.variants = tuple(variants)
        self._model = None
        self._proc = None
        self._device = None
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RealBackendUnavailable(
                f"real backend '{name}' needs torch (checkpoint '{checkpoint}'). "
                "Install `.[gpu]` and run on the GPU box; use a 'fake-*' encoder "
                "for Mac dev."
            ) from e

    # --- model loading (BRINGUP: validated on the GPU box) --------------------

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.family == "encodec":
            from transformers import EncodecModel

            self._model = EncodecModel.from_pretrained(self.checkpoint).eval().to(self._device)
        elif self.family == "wavlm":
            from transformers import AutoFeatureExtractor, WavLMModel

            self._model = WavLMModel.from_pretrained(self.checkpoint).eval().to(self._device)
            # Feature-extractor normalization matters: microsoft/wavlm-large ships
            # do_normalize=True; skipping it yields silently wrong-scale latents.
            self._proc = AutoFeatureExtractor.from_pretrained(self.checkpoint)
        else:
            raise NotImplementedError(
                f"'{self.family}' is a Phase-5 backend; wire and validate it at "
                "GPU-box bring-up (docs/GPU_BRINGUP.md)."
            )

    def _to_native(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import soxr

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr == self.native_sr:
            return wav
        return soxr.resample(wav, sr, self.native_sr).astype(np.float32)

    # --- encode ---------------------------------------------------------------

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, LatentResult]:
        import torch

        self._ensure()
        wav_n = self._to_native(wav, sr)
        n_native = wav_n.size
        out: dict[str, LatentResult] = {}
        with torch.no_grad():
            if self.family == "encodec":
                x = torch.from_numpy(wav_n).to(self._device)
                z = self._model.encoder(x[None, None, :])  # [1, D, T'] pre-quant
                frames = z[0].transpose(0, 1).float().cpu().numpy()
                fr = measure_frame_rate(frames.shape[0], n_native, self.native_sr)
                out["z"] = self._pack(frames, "z", fr)
            elif self.family == "wavlm":
                feats = self._proc(wav_n, sampling_rate=self.native_sr, return_tensors="pt")
                iv = feats.input_values.to(self._device)
                am = getattr(feats, "attention_mask", None)
                am = am.to(self._device) if am is not None else None
                hs = self._model(
                    iv, attention_mask=am, output_hidden_states=True
                ).hidden_states  # tuple[[1, T', D]]
                for v in self.variants:
                    frames = hs[_LAYER_MAP[v]][0].float().cpu().numpy()
                    fr = measure_frame_rate(frames.shape[0], n_native, self.native_sr)
                    out[v] = self._pack(frames, v, fr)
        return out

    def _pack(self, frames: np.ndarray, variant: str, fr: float) -> LatentResult:
        frames = np.asarray(frames, dtype=np.float64)
        meta = {
            "backend": "real",
            "name": self.name,
            "variant": variant,
            "native_sr": self.native_sr,
            "latent_dim": self.latent_dim,
            "n_frames": int(frames.shape[0]),
            "checkpoint": self.checkpoint,
            "device": self._device,
        }
        return LatentResult(frames=frames, frame_rate_hz=fr, meta=meta)
