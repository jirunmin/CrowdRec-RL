from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .env import (
    CandidateMode,
    CrowdRecEnv,
    EnvConfig,
    PROJECT_FEATURE_COLS,
    RewardMode,
    WORKER_FEATURE_COLS,
)


@dataclass(frozen=True)
class FastEnvConfig(EnvConfig):
    """Config for FastCrowdRecEnv."""

    processed_dir: str = "processed"
    stats_cache_filename: str = "fastenv_norm_cache.npz"


class FastCrowdRecEnv(CrowdRecEnv):
    """CrowdRecEnv subclass with train-stat normalization + disk cache.

    Keep all base env behaviors intact, only add:
    1) fit normalization stats from train split and cache to disk
    2) cache normalized worker/project features to avoid repeated z-score at runtime
    """

    def __init__(self, cfg: FastEnvConfig):
        self.fast_cfg = cfg
        self._worker_norm_map: Dict[int, np.ndarray] = {}
        self._project_norm_map: Dict[int, np.ndarray] = {}
        super().__init__(cfg)

    def _fit_normalizers(self) -> None:
        """Override base fitting: use train split stats + disk cache."""
        cache_path = os.path.join(self.fast_cfg.processed_dir, self.fast_cfg.stats_cache_filename)
        train_events_path = os.path.join(self.fast_cfg.processed_dir, "train_events.parquet")

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            self._w_mean = data["w_mean"].astype(np.float32)
            self._w_std = data["w_std"].astype(np.float32)
            self._p_mean = data["p_mean"].astype(np.float32)
            self._p_std = data["p_std"].astype(np.float32)

            worker_ids = data["worker_ids"].astype(np.int64)
            worker_norm = data["worker_norm"].astype(np.float32)
            self._worker_norm_map = {int(wid): worker_norm[i] for i, wid in enumerate(worker_ids)}

            project_ids = data["project_ids"].astype(np.int64)
            project_norm = data["project_norm"].astype(np.float32)
            self._project_norm_map = {int(pid): project_norm[i] for i, pid in enumerate(project_ids)}
            return

        if not os.path.exists(train_events_path):
            raise FileNotFoundError(f"train events not found for normalization cache: {train_events_path}")

        train_events = self._load_parquet(train_events_path)
        train_worker_ids = np.unique(train_events["worker"].to_numpy(dtype=np.int64, copy=False))
        train_project_ids = np.unique(train_events["project_id"].to_numpy(dtype=np.int64, copy=False))

        worker_block_df = self._worker_idx.reindex(train_worker_ids)
        worker_block = worker_block_df[list(WORKER_FEATURE_COLS)].to_numpy(dtype=np.float32, copy=True)
        worker_block = np.nan_to_num(worker_block, nan=0.0, posinf=0.0, neginf=0.0)

        project_block_df = self._project_idx.reindex(train_project_ids)
        project_block = project_block_df[list(PROJECT_FEATURE_COLS)].to_numpy(dtype=np.float32, copy=True)
        project_block = np.nan_to_num(project_block, nan=0.0, posinf=0.0, neginf=0.0)

        self._w_mean = worker_block.mean(axis=0)
        self._w_std = worker_block.std(axis=0)
        self._w_std[self._w_std == 0.0] = 1.0

        self._p_mean = project_block.mean(axis=0)
        self._p_std = project_block.std(axis=0)
        self._p_std[self._p_std == 0.0] = 1.0

        worker_norm = (worker_block - self._w_mean) / self._w_std
        project_norm = (project_block - self._p_mean) / self._p_std

        self._worker_norm_map = {int(wid): worker_norm[i] for i, wid in enumerate(train_worker_ids)}
        self._project_norm_map = {int(pid): project_norm[i] for i, pid in enumerate(train_project_ids)}

        os.makedirs(self.fast_cfg.processed_dir, exist_ok=True)
        np.savez(
            cache_path,
            w_mean=self._w_mean.astype(np.float32),
            w_std=self._w_std.astype(np.float32),
            p_mean=self._p_mean.astype(np.float32),
            p_std=self._p_std.astype(np.float32),
            worker_ids=train_worker_ids.astype(np.int64),
            worker_norm=worker_norm.astype(np.float32),
            project_ids=train_project_ids.astype(np.int64),
            project_norm=project_norm.astype(np.float32),
        )

    def _worker_vector(self, worker_id: int) -> np.ndarray:
        if self.fast_cfg.normalize_features:
            cached = self._worker_norm_map.get(int(worker_id))
            if cached is not None:
                return cached.copy()
        return super()._worker_vector(worker_id)

    def _project_block(self, project_ids: np.ndarray) -> np.ndarray:
        if not self.fast_cfg.normalize_features:
            return super()._project_block(project_ids)

        arr = np.zeros((len(project_ids), self.project_dim), dtype=np.float32)
        missing_idx = []
        missing_pid = []

        for i, pid in enumerate(project_ids.tolist()):
            cached = self._project_norm_map.get(int(pid))
            if cached is not None:
                arr[i] = cached
            else:
                missing_idx.append(i)
                missing_pid.append(int(pid))

        if missing_pid:
            fallback = super()._project_block(np.asarray(missing_pid, dtype=np.int64))
            arr[np.asarray(missing_idx, dtype=np.int64)] = fallback

        return arr


def make_fast_env(
    split: str = "train",
    processed_dir: str = "processed",
    reward_mode: RewardMode = "worker",
    candidate_mode: CandidateMode = "event_group",
    max_candidates: int = 20,
    seed: int = 42,
    normalize_features: bool = True,
    stats_cache_filename: str = "fastenv_norm_cache.npz",
) -> FastCrowdRecEnv:
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")

    cfg = FastEnvConfig(
        events_path=os.path.join(processed_dir, f"{split}_events.parquet"),
        worker_features_path=os.path.join(processed_dir, "worker_features.parquet"),
        project_features_path=os.path.join(processed_dir, "project_features.parquet"),
        candidates_path=os.path.join(processed_dir, "candidates.parquet"),
        reward_mode=reward_mode,
        candidate_mode=candidate_mode,
        max_candidates=max_candidates,
        seed=seed,
        normalize_features=normalize_features,
        processed_dir=processed_dir,
        stats_cache_filename=stats_cache_filename,
    )
    return FastCrowdRecEnv(cfg)


__all__ = [
    "FastCrowdRecEnv",
    "FastEnvConfig",
    "make_fast_env",
]
