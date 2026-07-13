"""RealCodecEncoder — lazy-loading adapters for the frozen backends (GPU box).

torch / transformers / dac are NOT installed on the Mac dev machine, so this
module is written but cannot be executed here. Constructing a real encoder on a
torch-less machine raises `RealBackendUnavailable`; downstream code falls back to
a `fake-*` encoder.

The encode paths target the models' documented APIs and are marked BRINGUP:
validated on the GPU box via docs/GPU_BRINGUP.md (shape + fake/real parity), NOT
on the Mac. Wired: EnCodec (continuous `z` + RVQ depths), WavLM (per layer, with
the model's own feature-extractor normalization), DAC (continuous `z` + RVQ
depths), Mimi (semantic vs acoustic streams — the §2.3 headline split). The
`LatentResult` contract is unchanged, so the pipeline is drop-in the moment the
box is reassembled.
"""

from __future__ import annotations

import numpy as np

from .base import LatentResult, measure_frame_rate

_LAYER_MAP = {"l1": 1, "l6": 6, "l12": 12, "l18": 18, "l24": 24}
# RVQ-depth variant name -> number of leading codebooks summed. `z` = continuous
# pre-quant latent (no quantization). Shared by the residual-codebook families.
_RVQ_DEPTHS = {"d1": 1, "d2": 2, "d4": 4, "d8": 8}


def _assert_depth_available(n_codebooks: int, depths: list[str], checkpoint: str) -> None:
    """numpy slicing `codes[:k]` silently returns min(k, n_q) codebooks, so a
    checkpoint exposing < 8 codebooks would cache a depth-6 latent under the label
    `d8`. Fail loud at bring-up instead of mislabelling (finding B2)."""
    need = max(_RVQ_DEPTHS[v] for v in depths)
    if n_codebooks < need:
        raise ValueError(
            f"{checkpoint} exposes {n_codebooks} RVQ codebooks but variant "
            f"'d{need}' needs {need}; drop the deep variants from REAL_SPECS or "
            "the requested depth would be silently truncated."
        )


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
        elif self.family == "dac":
            import dac

            self._model = dac.DAC.load(dac.utils.download(model_type="44khz")).eval().to(self._device)
        elif self.family == "mimi":
            from transformers import MimiModel

            self._model = MimiModel.from_pretrained(self.checkpoint).eval().to(self._device)
        else:
            raise NotImplementedError(
                f"'{self.family}' has no wired backend; add it and validate at "
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
        with torch.no_grad():
            if self.family == "encodec":
                by_variant = self._encode_encodec(wav_n)
            elif self.family == "wavlm":
                by_variant = self._encode_wavlm(wav_n)
            elif self.family == "dac":
                by_variant = self._encode_dac(wav_n)
            elif self.family == "mimi":
                by_variant = self._encode_mimi(wav_n)
            else:  # unreachable (`_ensure` already raised) — explicit so a future
                raise NotImplementedError(self.family)  # 5th family can't UnboundLocalError (S5)
        return {
            v: self._pack(frames, v, measure_frame_rate(frames.shape[0], wav_n.size, self.native_sr))
            for v, frames in by_variant.items()
        }

    # A residual codec exposes `z` (continuous pre-quant) and `d{k}` = the sum of
    # the first k RVQ codebook embeddings. Both live in the same [T, D] embedding
    # space; only the requested variants are materialised.
    def _encode_encodec(self, wav_n: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        x = torch.from_numpy(wav_n).to(self._device)[None, None, :]
        z = self._model.encoder(x)  # [1, D, T'] continuous, pre-quant
        out: dict[str, np.ndarray] = {}
        if "z" in self.variants:
            out["z"] = z[0].transpose(0, 1).float().cpu().numpy()
        depths = [v for v in self.variants if v in _RVQ_DEPTHS]
        if depths:
            codes = self._model.quantizer.encode(z, self._model.config.target_bandwidths[-1])  # [n_q, 1, T']
            _assert_depth_available(codes.shape[0], depths, self.checkpoint)  # else d8 silently becomes d6 (W B2)
            for v in depths:
                zq = self._model.quantizer.decode(codes[: _RVQ_DEPTHS[v]])  # sum of first k codebooks -> [1, D, T']
                out[v] = zq[0].transpose(0, 1).float().cpu().numpy()
        return out

    def _encode_wavlm(self, wav_n: np.ndarray) -> dict[str, np.ndarray]:
        feats = self._proc(wav_n, sampling_rate=self.native_sr, return_tensors="pt")
        iv = feats.input_values.to(self._device)
        am = getattr(feats, "attention_mask", None)
        am = am.to(self._device) if am is not None else None
        hs = self._model(iv, attention_mask=am, output_hidden_states=True).hidden_states  # tuple[[1, T', D]]
        return {v: hs[_LAYER_MAP[v]][0].float().cpu().numpy() for v in self.variants}

    def _encode_dac(self, wav_n: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        x = torch.from_numpy(wav_n).to(self._device)[None, None, :]
        x = self._model.preprocess(x, self.native_sr)
        z, codes, _, _, _ = self._model.encode(x)  # z:[1, D, T'] quantized, codes:[1, n_q, T']
        out: dict[str, np.ndarray] = {}
        if "z" in self.variants:
            out["z"] = z[0].transpose(0, 1).float().cpu().numpy()
        depths = [v for v in self.variants if v in _RVQ_DEPTHS]
        if depths:
            _assert_depth_available(codes.shape[1], depths, self.checkpoint)  # else d8 silently becomes d6 (W B2)
            for v in depths:
                zq = self._model.quantizer.from_codes(codes[:, : _RVQ_DEPTHS[v]])[0]  # [1, D, T']
                out[v] = zq[0].transpose(0, 1).float().cpu().numpy()
        return out

    # Mimi's headline split (§2.3): codebook 0 is the WavLM-distilled *semantic*
    # stream; the residual codebooks are the *acoustic* stream. We surface each as
    # its own dequantized embedding, probed separately.
    def _encode_mimi(self, wav_n: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        x = torch.from_numpy(wav_n).to(self._device)[None, None, :]
        codes = self._model.encode(x).audio_codes  # [1, n_q, T']
        q = self._model.quantizer
        streams = {
            "semantic": q.semantic_residual_vector_quantizer.decode(codes[:, :1]),
            "acoustic": q.acoustic_residual_vector_quantizer.decode(codes[:, 1:]),
        }
        return {v: streams[v][0].transpose(0, 1).float().cpu().numpy() for v in self.variants}

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
