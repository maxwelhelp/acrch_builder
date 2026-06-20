from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


DEFAULT_CLASSES = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")


@dataclass
class AudioBatch:
    x: torch.Tensor
    y: torch.Tensor

    @property
    def waveforms(self) -> torch.Tensor:
        return self.x


def _import_speechcommands():
    try:
        import torchaudio
        from torchaudio.datasets import SPEECHCOMMANDS
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "torchaudio is required for SpeechCommands. Install a matching torchaudio build."
        ) from e
    return torchaudio, SPEECHCOMMANDS


def normalize_classes(classes: Sequence[str] | str | None) -> List[str]:
    if classes is None:
        return list(DEFAULT_CLASSES)
    if isinstance(classes, str):
        raw = classes.replace(";", ",").split(",")
        return [item.strip() for item in raw if item.strip()]
    return [str(item).strip() for item in classes if str(item).strip()]


def collate_waveforms(
    batch: Sequence[tuple[torch.Tensor, int] | tuple[torch.Tensor, int, int]],
    seconds: float = 1.0,
    sample_rate: int = 16000,
) -> AudioBatch:
    target_len = int(seconds * sample_rate)
    wavs: List[torch.Tensor] = []
    ys: List[int] = []
    for item in batch:
        if len(item) == 2:
            wav, y = item  # type: ignore[misc]
            sr = sample_rate
        else:
            wav, sr, y = item  # type: ignore[misc]
        wav = wav.float()
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        if sr != sample_rate:
            torchaudio, _ = _import_speechcommands()
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, sample_rate).squeeze(0)
        if wav.numel() < target_len:
            wav = F.pad(wav, (0, target_len - wav.numel()))
        elif wav.numel() > target_len:
            wav = wav[:target_len]
        wavs.append(wav)
        ys.append(int(y))
    return AudioBatch(x=torch.stack(wavs, dim=0), y=torch.tensor(ys, dtype=torch.long))


class SpeechCommandsFiltered(torch.utils.data.Dataset):
    """Fast filtered wrapper around torchaudio.datasets.SPEECHCOMMANDS."""

    def __init__(
        self,
        root: str,
        subset: str,
        classes: Sequence[str] | str | None = None,
        download: bool = False,
        limit: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.classes = normalize_classes(classes)
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        self.subset = subset
        self._torchaudio, SPEECHCOMMANDS = _import_speechcommands()

        root_path = Path(root).expanduser().resolve()
        if download:
            root_path.mkdir(parents=True, exist_ok=True)
        elif not root_path.exists():
            raise RuntimeError(f"SpeechCommands root does not exist: {root_path}")

        try:
            self.base = SPEECHCOMMANDS(root=str(root_path), download=download, subset=subset)
        except Exception as e:  # pragma: no cover - external dataset init
            raise RuntimeError(
                f"Failed to initialize/download SpeechCommands at {root_path}. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        walker = getattr(self.base, "_walker", None)
        if walker is None:
            raise RuntimeError("torchaudio SPEECHCOMMANDS internals changed: _walker not found")

        idxs: List[int] = []
        for i, file_path in enumerate(walker):
            label = Path(str(file_path)).parent.name
            if label in self.class_to_id:
                idxs.append(i)

        rng = random.Random(seed + {"training": 0, "validation": 1, "testing": 2}.get(subset, 3))
        rng.shuffle(idxs)
        if limit and limit > 0:
            idxs = idxs[:limit]
        self.idxs = idxs

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        waveform, sample_rate, label, *_ = self.base[self.idxs[idx]]
        if sample_rate != 16000:
            waveform = self._torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        waveform = waveform.squeeze(0).float()
        return waveform, sample_rate, self.class_to_id[label]


class SpeechCommandsAcceptanceTask:
    """Training/evaluation wrapper for SpeechCommands acceptance smoke."""

    def __init__(
        self,
        root: str,
        classes: Sequence[str] | str | None = None,
        download: bool = False,
        train_limit: int = 0,
        val_limit: int = 0,
        test_limit: int = 0,
        seed: int = 0,
        seconds: float = 1.0,
    ) -> None:
        self.root = str(root)
        self.classes = normalize_classes(classes)
        self.download = download
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit
        self.seed = seed
        self.seconds = seconds
        self.sample_rate = 16000
        self.train_ds = SpeechCommandsFiltered(
            root=self.root,
            subset="training",
            classes=self.classes,
            download=download,
            limit=train_limit,
            seed=seed,
        )
        self.val_ds = SpeechCommandsFiltered(
            root=self.root,
            subset="validation",
            classes=self.classes,
            download=False,
            limit=val_limit,
            seed=seed,
        )
        self.test_ds = None
        if test_limit != 0:
            self.test_ds = SpeechCommandsFiltered(
                root=self.root,
                subset="testing",
                classes=self.classes,
                download=False,
                limit=test_limit,
                seed=seed,
            )

    def loaders(
        self,
        batch_size: int,
        eval_batch_size: int,
        workers: int,
        pin_memory: bool,
        drop_last: bool,
    ):
        collate = lambda b: collate_waveforms(b, seconds=self.seconds, sample_rate=self.sample_rate)
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate,
            persistent_workers=(workers > 0),
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate,
            persistent_workers=(workers > 0),
        )
        test_loader = None
        if self.test_ds is not None:
            test_loader = torch.utils.data.DataLoader(
                self.test_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_memory,
                drop_last=False,
                collate_fn=collate,
                persistent_workers=(workers > 0),
            )
        return train_loader, val_loader, test_loader

    def _split_dataset(self, split: str):
        if split == "training":
            return self.train_ds
        if split == "validation":
            return self.val_ds
        if split == "testing":
            return self.test_ds or self.val_ds
        raise ValueError(f"unknown SpeechCommands split: {split!r}")

    def sample(self, batch_size: int, device: str | torch.device, split: str = "validation") -> AudioBatch:
        ds = self._split_dataset(split)
        if ds is None:
            raise RuntimeError(f"SpeechCommands split {split!r} is unavailable")
        idxs = torch.randint(0, len(ds), (batch_size,))
        batch = [ds[int(i)] for i in idxs.tolist()]
        out = collate_waveforms(batch, seconds=self.seconds, sample_rate=self.sample_rate)
        return AudioBatch(x=out.x.to(device), y=out.y.to(device))
