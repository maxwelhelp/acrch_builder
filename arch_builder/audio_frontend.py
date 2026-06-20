from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ActionMatrixModel


FrontendName = Literal["raw", "conv", "structured"]


@dataclass
class AudioBatch:
    waveforms: torch.Tensor
    y: torch.Tensor


def _dct_basis(frame_size: int, bins: int, device=None, dtype=None) -> torch.Tensor:
    n = torch.arange(frame_size, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(bins, device=device, dtype=dtype).unsqueeze(0)
    basis = torch.cos(math.pi / frame_size * (n + 0.5) * k)
    scale = torch.ones(bins, device=device, dtype=dtype) * math.sqrt(2.0 / frame_size)
    scale[0] = math.sqrt(1.0 / frame_size)
    return basis * scale


class SyntheticAudioOrderTask:
    """Tiny audio-order dataset used for the structured frontend proof.

    The classes share the same tone inventory and differ only in order.
    Raw global pooling loses that ordering signal.
    """

    def __init__(
        self,
        length: int = 256,
        noise_std: float = 0.035,
        burst_width: int = 28,
        low_freq: float = 3.5,
        high_freq: float = 11.5,
    ) -> None:
        self.length = length
        self.noise_std = noise_std
        self.burst_width = burst_width
        self.low_freq = low_freq
        self.high_freq = high_freq

        self._time = torch.linspace(0.0, 1.0, length)
        self._window1 = self._make_window(0.18, burst_width)
        self._window2 = self._make_window(0.70, burst_width)

    def _make_window(self, center: float, width: int) -> torch.Tensor:
        t = self._time
        span = max(1, width) / float(self.length)
        dist = (t - center).abs() / span
        window = (1.0 - dist).clamp_min(0.0)
        return window

    def sample(self, batch_size: int, device: str | torch.device) -> AudioBatch:
        time = self._time.to(device=device)
        window1 = self._window1.to(device=device)
        window2 = self._window2.to(device=device)
        labels = torch.randint(0, 2, (batch_size,), device=device)

        low_freq = self.low_freq + 0.25 * torch.rand(batch_size, device=device)
        high_freq = self.high_freq + 0.50 * torch.rand(batch_size, device=device)
        phase_a = 2.0 * math.pi * torch.rand(batch_size, device=device)
        phase_b = 2.0 * math.pi * torch.rand(batch_size, device=device)
        drift = 0.12 * torch.sin(2.0 * math.pi * (1.0 + 0.15 * torch.rand(batch_size, device=device))[:, None] * time[None, :])

        low_wave = torch.sin(2.0 * math.pi * low_freq[:, None] * time[None, :] + phase_a[:, None])
        high_wave = torch.sin(2.0 * math.pi * high_freq[:, None] * time[None, :] + phase_b[:, None])
        common = 0.08 * torch.sin(2.0 * math.pi * 6.0 * time[None, :] + 0.5 * phase_a[:, None])

        first = window1[None, :] * low_wave + window2[None, :] * high_wave
        second = window1[None, :] * high_wave + window2[None, :] * low_wave
        waveforms = torch.where(labels[:, None] == 0, first, second)
        waveforms = waveforms + common + drift + self.noise_std * torch.randn_like(waveforms)
        return AudioBatch(waveforms=waveforms, y=labels)


class AudioFrontendBase(nn.Module):
    frontend_name: FrontendName

    def report(self, waveform_length: int, slots: int, dim: int) -> Dict[str, float]:
        params = float(sum(p.numel() for p in self.parameters()))
        flops = float(self.estimate_flops(waveform_length, slots, dim))
        memory = float(self.estimate_activation_bytes(waveform_length, slots, dim))
        return {
            "params": params,
            "flops": flops,
            "activation_bytes": memory,
        }

    def estimate_flops(self, waveform_length: int, slots: int, dim: int) -> int:
        raise NotImplementedError

    def estimate_activation_bytes(self, waveform_length: int, slots: int, dim: int) -> int:
        raise NotImplementedError


class RawBasicAudioFrontend(AudioFrontendBase):
    frontend_name: FrontendName = "raw"

    def __init__(self, slots: int, dim: int) -> None:
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.proj = nn.Linear(4, dim)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError(f"expected waveforms shaped [batch, time], got {tuple(waveforms.shape)}")
        mean = waveforms.mean(dim=-1, keepdim=True)
        abs_mean = waveforms.abs().mean(dim=-1, keepdim=True)
        energy = waveforms.pow(2).mean(dim=-1, keepdim=True)
        diff = F.pad((waveforms[:, 1:] - waveforms[:, :-1]).abs().mean(dim=-1, keepdim=True), (0, 0))
        stats = torch.cat([mean, abs_mean, energy, diff], dim=-1)
        base = self.proj(stats)
        return base[:, None, :].expand(-1, self.slots, -1)

    def estimate_flops(self, waveform_length: int, slots: int, dim: int) -> int:
        return int(waveform_length * 8 + 4 * dim)

    def estimate_activation_bytes(self, waveform_length: int, slots: int, dim: int) -> int:
        return int(4 * (waveform_length + dim + slots * dim))


class ConvScaffoldAudioFrontend(AudioFrontendBase):
    frontend_name: FrontendName = "conv"

    def __init__(self, slots: int, dim: int) -> None:
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(12, 24, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(slots)
        self.proj = nn.Linear(24, dim)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError(f"expected waveforms shaped [batch, time], got {tuple(waveforms.shape)}")
        x = self.conv(waveforms.unsqueeze(1))
        x = self.pool(x).transpose(1, 2)
        return self.proj(x)

    def estimate_flops(self, waveform_length: int, slots: int, dim: int) -> int:
        conv1 = waveform_length * 1 * 12 * 7 // 2
        conv2 = (waveform_length // 2) * 12 * 24 * 5 // 2
        proj = slots * 24 * dim
        return int(conv1 + conv2 + proj)

    def estimate_activation_bytes(self, waveform_length: int, slots: int, dim: int) -> int:
        return int(4 * (waveform_length + (waveform_length // 2) * 12 + slots * 24 + slots * dim))


class StructuredMatrixAudioFrontend(AudioFrontendBase):
    frontend_name: FrontendName = "structured"

    def __init__(
        self,
        slots: int,
        dim: int,
        frame_size: int = 32,
        hop_size: int = 16,
        dct_bins: int = 10,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.dct_bins = dct_bins
        feature_dim = 1 + 1 + 1 + dct_bins
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, dim)
        self.register_buffer("window", torch.hann_window(frame_size), persistent=False)
        self.register_buffer("dct_basis", _dct_basis(frame_size, dct_bins), persistent=False)

    def _frame_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        frames = waveforms.unfold(-1, self.frame_size, self.hop_size)
        window = self.window.to(dtype=waveforms.dtype, device=waveforms.device)
        dct_basis = self.dct_basis.to(dtype=waveforms.dtype, device=waveforms.device)
        windowed = frames * window
        energy = windowed.pow(2).mean(dim=-1, keepdim=True)
        delta = F.pad(energy[:, 1:] - energy[:, :-1], (0, 0, 1, 0))
        onset = F.relu(delta)
        dct = torch.matmul(windowed, dct_basis)
        spectral = torch.log1p(dct.abs())
        spectral = spectral[..., : self.dct_bins]
        feats = torch.cat([energy, delta, onset, spectral], dim=-1)
        return self.feature_norm(feats)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError(f"expected waveforms shaped [batch, time], got {tuple(waveforms.shape)}")
        feats = self._frame_features(waveforms)
        pooled = F.adaptive_avg_pool1d(feats.transpose(1, 2), self.slots).transpose(1, 2)
        return self.proj(pooled)

    def estimate_flops(self, waveform_length: int, slots: int, dim: int) -> int:
        frames = max(1, 1 + (waveform_length - self.frame_size) // self.hop_size)
        dct = frames * self.frame_size * self.dct_bins
        feat = frames * (1 + 1 + 1 + self.dct_bins)
        proj = frames * (1 + 1 + 1 + self.dct_bins) * dim
        return int(dct + feat + proj)

    def estimate_activation_bytes(self, waveform_length: int, slots: int, dim: int) -> int:
        frames = max(1, 1 + (waveform_length - self.frame_size) // self.hop_size)
        feature_dim = 1 + 1 + 1 + self.dct_bins
        return int(4 * (frames * self.frame_size + frames * feature_dim + slots * dim))


def build_frontend(variant: FrontendName, slots: int, dim: int) -> AudioFrontendBase:
    if variant == "raw":
        return RawBasicAudioFrontend(slots=slots, dim=dim)
    if variant == "conv":
        return ConvScaffoldAudioFrontend(slots=slots, dim=dim)
    if variant == "structured":
        return StructuredMatrixAudioFrontend(slots=slots, dim=dim)
    raise ValueError(f"unknown frontend variant: {variant!r}")


class AudioMatrixClassifier(nn.Module):
    def __init__(
        self,
        frontend: AudioFrontendBase,
        dim: int,
        slots: int,
        layers: int = 1,
        classes: int = 2,
        top_k: int = 8,
        sim_rank: int = 8,
        input_norm: str = "none",
        state_norm: str = "none",
        final_read: str = "last",
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = ActionMatrixModel(
            dim=dim,
            slots=slots,
            layers=layers,
            classes=classes,
            top_k=top_k,
            sim_rank=sim_rank,
            input_norm=input_norm,
            state_norm=state_norm,
            final_read=final_read,
        )

    def forward(self, waveforms: torch.Tensor, tau: float = 1.0, curriculum_mode: str = "teacher"):
        features = self.frontend(waveforms)
        logits, trace = self.backbone(features, tau=tau, curriculum_mode=curriculum_mode)
        trace = dict(trace)
        trace["frontend_variant"] = self.frontend.frontend_name
        trace["frontend_shape"] = tuple(features.shape)
        return logits, trace

    def report(self, waveform_length: int, slots: int, dim: int) -> Dict[str, float]:
        frontend_report = self.frontend.report(waveform_length, slots, dim)
        backbone_params = float(sum(p.numel() for p in self.backbone.parameters()))
        total_params = float(sum(p.numel() for p in self.parameters()))
        return {
            "frontend_params": frontend_report["params"],
            "frontend_flops": frontend_report["flops"],
            "frontend_activation_bytes": frontend_report["activation_bytes"],
            "backbone_params": backbone_params,
            "total_params": total_params,
            "total_flops": frontend_report["flops"] + float(self.backbone.num_layers * slots * dim * 128),
            "total_activation_bytes": frontend_report["activation_bytes"] + float(slots * dim * 4),
        }
