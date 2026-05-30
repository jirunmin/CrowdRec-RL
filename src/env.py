"""
RL environment for CrowdRec-RL.

The environment replays the offline event stream produced by ``src/event_stream.py``
and serves it to a DQN-style agent in two modes:

candidate_mode = "event_group"
    Each step's candidate set comes from the *(worker, timestamp)* group already
    present in the event split. Typically the group is 1 positive + N negatives
    (default N=2). Reward for every candidate is precomputed in the parquet.
    Used for both training and primary offline evaluation.

candidate_mode = "top_k"
    Each step's candidate set comes from ``candidates.parquet`` (the Top-20
    active projects at that timestamp). Only the ground-truth project has a
    known reward; all others get 0. Used for stricter counterfactual ranking.

The state is action-conditioned: ``reset()`` / ``step()`` returns
{
    "worker_state":   shape (D_w,)         # static per-worker features
    "candidate_state": shape (n_cand, D_p+D_m)  # per-candidate project + match feats
    "valid_mask":     shape (n_cand,) bool
    "info":           dict with event_id, worker, candidate project_ids, ...
}

Action = integer index into the candidate set.

Reward is selected by ``reward_mode`` ∈ {"worker", "requester"}.

The env is offline / counterfactual: it does not simulate worker behavior;
when the agent picks a candidate it receives the precomputed reward for the
(worker, project) pair, and 0 if that pair was not observed.

Designed to support the rest of the project (C: basic DQN, D: Double/Dueling DQN,
E: baselines), so it stays minimal and dependency-light (NumPy/Pandas only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .reward import (
    REQUESTER_COEF,
    RequesterRewardCoef,
    RewardMode,
    WORKER_COEF,
    WorkerRewardCoef,
    pick_precomputed_column,
    requester_reward,
    worker_reward,
)


CandidateMode = Literal["event_group", "top_k"]


# ---------------------------------------------------------------------------
# Feature column lists – kept in one place so that downstream models share them.
# ---------------------------------------------------------------------------

WORKER_FEATURE_COLS: Tuple[str, ...] = (
    "worker_quality",
    "worker_total_entries",
    "worker_total_projects",
    "worker_win_count",
    "worker_finalist_count",
    "worker_avg_score",
    "worker_win_rate",
    "worker_active_days",
    "worker_pref_category",
    "worker_pref_sub_category",
    "worker_pref_industry",
    "worker_category_entropy",
)

PROJECT_FEATURE_COLS: Tuple[str, ...] = (
    "project_category",
    "project_sub_category",
    "project_industry_code",
    "project_entry_count",
    "project_total_awards",
    "project_duration_days",
    "project_is_featured",
    "project_average_score",
    "project_worker_count",
    "project_has_winner",
)

MATCH_FEATURE_COLS: Tuple[str, ...] = (
    "match_category",
    "match_sub_category",
    "match_industry",
    "match_quality_gap",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    """Configuration for ``CrowdRecEnv``.

    Attributes:
        events_path:    Parquet file produced by the preprocessing pipeline.
                        Contains all (positive + negative) events for the split.
        worker_features_path: Worker static feature parquet.
        project_features_path: Project static feature parquet.
        candidates_path: Deprecated – candidates are now embedded in events parquet.
                         ``candidate_mode="top_k"``.
        reward_mode:    "worker" | "requester".
        candidate_mode: "event_group" | "top_k".
        max_candidates: Pad / clip candidate count. The valid_mask hides padding.
        seed:           RNG seed for tie-breaking and the random baseline.
        normalize_features: If True, z-score numeric features using statistics
                         computed lazily on first reset. Disabled by default to
                         keep the env deterministic for unit tests.
    """

    events_path: str
    worker_features_path: str
    project_features_path: str
    candidates_path: Optional[str] = None
    reward_mode: RewardMode = "worker"
    candidate_mode: CandidateMode = "event_group"
    max_candidates: int = 20
    seed: int = 42
    normalize_features: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CrowdRecEnv:
    """Offline replay environment for crowdsourcing task recommendation.

    Lightweight Gym-style API:
        obs = env.reset()
        obs, reward, done, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        if cfg.candidate_mode not in ("event_group", "top_k"):
            raise ValueError(f"candidate_mode must be 'event_group' or 'top_k', got {cfg.candidate_mode!r}")
        if cfg.reward_mode not in ("worker", "requester"):
            raise ValueError(f"reward_mode must be 'worker' or 'requester', got {cfg.reward_mode!r}")

        # --- Load data ---
        self.events_df: pd.DataFrame = self._load_parquet(cfg.events_path)
        self.worker_features: pd.DataFrame = self._load_parquet(cfg.worker_features_path)
        self.project_features: pd.DataFrame = self._load_parquet(cfg.project_features_path)

        # Reconcile worker_quality with the value A's reward pipeline actually used.
        # A keeps `worker_quality = -1` in worker_features.parquet for unlabeled
        # workers (~85% of the table), but writes the ML-predicted quality
        # (`worker_quality_pred`) into the event rows and rewards. If we read
        # `worker_quality` straight from the feature table, the env state would
        # report -1 for those workers and match_quality_gap would diverge from
        # the parquet match_quality_gap column. Prefer `worker_quality_pred`
        # whenever it is available.
        if "worker_quality_pred" in self.worker_features.columns:
            pred = self.worker_features["worker_quality_pred"]
            raw = self.worker_features["worker_quality"]
            # Use predicted value where raw is missing (< 0 = unlabeled).
            self.worker_features = self.worker_features.assign(
                worker_quality=raw.where(raw >= 0, pred)
            )

        # Index lookup tables for fast feature retrieval.
        self._worker_idx = self.worker_features.set_index("worker_id")
        self._project_idx = self.project_features.set_index("project_id")

        if cfg.candidate_mode == "top_k":
            # candidates are embedded in events_df as candidate_projects column
            pos = self.events_df[self.events_df["label"] == 1]
            self._cand_lookup = dict(zip(pos["event_id"], pos["candidate_projects"]))
        else:
            self.candidates_df = pd.DataFrame()
            self._cand_lookup = {}

        # Build the step plan – list of episodes/steps determined by mode.
        self._steps: List[Dict[str, Any]] = self._build_steps()
        self._cursor: int = 0

        # Cache feature dimensions.
        self.worker_dim: int = len(WORKER_FEATURE_COLS)
        self.project_dim: int = len(PROJECT_FEATURE_COLS)
        self.match_dim: int = len(MATCH_FEATURE_COLS)
        self.candidate_dim: int = self.project_dim + self.match_dim
        self.max_candidates: int = cfg.max_candidates

        # Optional normalization.
        if cfg.normalize_features:
            self._fit_normalizers()
        else:
            self._w_mean = self._w_std = None
            self._p_mean = self._p_std = None

    # ------------------------------------------------------------------
    # Public Gym-style API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._steps)

    def reset(self) -> Dict[str, Any]:
        """Reset cursor to the first step. Returns the initial observation."""
        self._cursor = 0
        return self._build_obs(self._steps[self._cursor])

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply ``action``, advance one step, return ``(obs, reward, done, info)``.

        ``action`` is an integer index into the current candidate list (0..n_cand-1).
        Selecting an out-of-range or padded index yields reward=0.
        """
        if self._cursor >= len(self._steps):
            raise RuntimeError("Episode already finished – call reset().")

        step = self._steps[self._cursor]
        reward, info_step = self._compute_reward(step, action)

        self._cursor += 1
        done = self._cursor >= len(self._steps)

        next_obs: Dict[str, Any]
        if done:
            next_obs = self._terminal_obs()
        else:
            next_obs = self._build_obs(self._steps[self._cursor])

        info = {
            "event_id": step["event_id"],
            "worker": step["worker"],
            "timestamp": step["timestamp"],
            "ground_truth_index": step["gt_index"],
            "ground_truth_project": step["gt_project"],
            "candidate_projects": step["candidate_projects"],
            "selected_index": int(action),
            "reward_mode": self.cfg.reward_mode,
            "hit": int(action == step["gt_index"]),
            **info_step,
        }
        return next_obs, float(reward), done, info

    def random_policy(self, obs: Dict[str, Any]) -> int:
        """Uniformly sample a valid action from the current observation."""
        mask = obs["valid_mask"]
        valid_idx = np.flatnonzero(mask)
        if valid_idx.size == 0:
            return 0
        return int(self._rng.choice(valid_idx))

    @property
    def state_dim(self) -> int:
        """Total flat state size (worker + max_candidates * per-candidate)."""
        return self.worker_dim + self.max_candidates * self.candidate_dim

    # ------------------------------------------------------------------
    # Step construction
    # ------------------------------------------------------------------

    def _build_steps(self) -> List[Dict[str, Any]]:
        """Pre-compute the full sequence of steps for the configured mode."""
        if self.cfg.candidate_mode == "event_group":
            return self._build_steps_event_group()
        return self._build_steps_top_k()

    def _build_steps_event_group(self) -> List[Dict[str, Any]]:
        """One step per (worker, timestamp) group; candidates = group rows.

        Within each group we deterministically shuffle the candidate order so
        that the positive sample is NOT systematically at index 0. Otherwise a
        DQN can exploit the dataset by always picking action 0 and reach a
        spuriously high hit rate.
        """
        # NOTE: do NOT sort by `label` — that biases gt_index to 0.
        df = self.events_df.sort_values(["timestamp", "worker"],
                                         ascending=[True, True])
        steps: List[Dict[str, Any]] = []

        reward_col = pick_precomputed_column(self.cfg.reward_mode)

        # Independent RNG seeded from cfg.seed so shuffles are reproducible
        # across runs and unaffected by the random_policy RNG.
        shuffle_rng = np.random.default_rng(self.cfg.seed)

        # groupby preserves order due to sort_values above.
        rng = np.random.default_rng(self.cfg.seed)
        for (worker, timestamp), group in df.groupby(["worker", "timestamp"], sort=False):
            cand_pids = group["project_id"].to_numpy()
            labels = group["label"].to_numpy(dtype=np.int8)
            rewards = group[reward_col].to_numpy(dtype=np.float32)

            # Per-group permutation – deterministic given cfg.seed.
            perm = shuffle_rng.permutation(len(cand_pids))
            cand_pids = cand_pids[perm]
            labels = labels[perm]
            rewards = rewards[perm]

            # Ground truth = the row with label==1 (there is at most one in 99.999% of cases).
            pos_indices = np.flatnonzero(labels == 1)
            gt_index = int(pos_indices[0]) if pos_indices.size > 0 else -1
            gt_project = int(cand_pids[gt_index]) if gt_index >= 0 else -1

            steps.append({
                "event_id": int(group["event_id"].iloc[0]),
                "worker": int(worker),
                "timestamp": timestamp,
                "candidate_projects": cand_pids.astype(np.int64),
                "candidate_rewards": rewards,
                "candidate_labels": labels,
                "gt_index": gt_index,
                "gt_project": gt_project,
            })
        return steps

    def _build_steps_top_k(self) -> List[Dict[str, Any]]:
        """One step per positive event; candidates = Top-K active at that time."""
        reward_col = pick_precomputed_column(self.cfg.reward_mode)
        pos = self.events_df[self.events_df["label"] == 1].sort_values("timestamp")

        steps: List[Dict[str, Any]] = []
        for _, row in pos.iterrows():
            event_id = int(row["event_id"])
            cand_pids = self._cand_lookup.get(event_id)
            if cand_pids is None or len(cand_pids) == 0:
                # No active candidates at this time – skip rather than fabricate.
                continue
            cand_pids = np.asarray(cand_pids, dtype=np.int64)

            true_pid = int(row["project_id"])
            true_reward = float(row[reward_col])

            rewards = np.zeros(len(cand_pids), dtype=np.float32)
            labels = np.zeros(len(cand_pids), dtype=np.int8)

            gt_pos = np.flatnonzero(cand_pids == true_pid)
            if gt_pos.size > 0:
                rewards[gt_pos[0]] = true_reward
                labels[gt_pos[0]] = 1
                gt_index = int(gt_pos[0])
            else:
                # The chosen project was not in the Top-K candidate set: in this
                # mode the agent cannot recover the reward. Leave gt_index = -1
                # so callers know hit-rate is undefined for this step.
                gt_index = -1

            steps.append({
                "event_id": event_id,
                "worker": int(row["worker"]),
                "timestamp": row["timestamp"],
                "candidate_projects": cand_pids,
                "candidate_rewards": rewards,
                "candidate_labels": labels,
                "gt_index": gt_index,
                "gt_project": true_pid,
            })
        return steps

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, step: Dict[str, Any], action: int) -> Tuple[float, Dict[str, Any]]:
        rewards = step["candidate_rewards"]
        n = rewards.shape[0]
        if action < 0 or action >= n:
            return 0.0, {"action_valid": False}
        return float(rewards[action]), {"action_valid": True}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self, step: Dict[str, Any]) -> Dict[str, Any]:
        worker_vec = self._worker_vector(step["worker"]).astype(np.float32)

        cand_pids: np.ndarray = step["candidate_projects"]
        n_cand = min(len(cand_pids), self.max_candidates)
        cand_state = np.zeros((self.max_candidates, self.candidate_dim), dtype=np.float32)
        valid_mask = np.zeros(self.max_candidates, dtype=bool)

        if n_cand > 0:
            project_block = self._project_block(cand_pids[:n_cand])
            match_block = self._match_block(step["worker"], cand_pids[:n_cand])
            cand_state[:n_cand] = np.concatenate([project_block, match_block], axis=1)
            valid_mask[:n_cand] = True

        return {
            "worker_state": worker_vec,
            "candidate_state": cand_state,
            "valid_mask": valid_mask,
            "info": {
                "event_id": step["event_id"],
                "worker": step["worker"],
                "timestamp": step["timestamp"],
                "candidate_projects": cand_pids,
                "n_valid": int(n_cand),
            },
        }

    def _terminal_obs(self) -> Dict[str, Any]:
        """All-zero terminal observation, returned with done=True."""
        return {
            "worker_state": np.zeros(self.worker_dim, dtype=np.float32),
            "candidate_state": np.zeros((self.max_candidates, self.candidate_dim), dtype=np.float32),
            "valid_mask": np.zeros(self.max_candidates, dtype=bool),
            "info": {"terminal": True},
        }

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _worker_vector(self, worker_id: int) -> np.ndarray:
        try:
            row = self._worker_idx.loc[worker_id]
        except KeyError:
            return np.zeros(self.worker_dim, dtype=np.float32)
        if isinstance(row, pd.DataFrame):  # duplicate worker_ids edge case
            row = row.iloc[0]
        vec = row[list(WORKER_FEATURE_COLS)].to_numpy(dtype=np.float32, copy=True)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if self._w_mean is not None:
            vec = (vec - self._w_mean) / self._w_std
        return vec

    def _project_block(self, project_ids: np.ndarray) -> np.ndarray:
        # Reindex returns NaN for missing ids; we map to 0.
        try:
            block = self._project_idx.reindex(project_ids)[list(PROJECT_FEATURE_COLS)]
        except KeyError:
            block = pd.DataFrame(0.0, index=project_ids, columns=list(PROJECT_FEATURE_COLS))
        arr = block.to_numpy(dtype=np.float32, copy=True)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self._p_mean is not None:
            arr = (arr - self._p_mean) / self._p_std
        return arr

    def _match_block(self, worker_id: int, project_ids: np.ndarray) -> np.ndarray:
        # Pull worker preference fields. NOTE: do NOT use ``x or default`` here —
        # ``0.0 or -1`` evaluates to -1 in Python (0.0 is falsy), which would
        # silently break match_industry for the ~37% of workers whose
        # pref_industry happens to be the integer code 0 (e.g. healthcare).
        # Always check pd.isna explicitly.
        try:
            w_row = self._worker_idx.loc[worker_id]
            if isinstance(w_row, pd.DataFrame):
                w_row = w_row.iloc[0]

            def _coerce(v: Any, default: float) -> float:
                if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                    return default
                return float(v)

            w_pref_cat = _coerce(w_row.get("worker_pref_category", -1), -1.0)
            w_pref_sub = _coerce(w_row.get("worker_pref_sub_category", -1), -1.0)
            w_pref_ind = _coerce(w_row.get("worker_pref_industry", -1), -1.0)
            w_q = _coerce(w_row.get("worker_quality", 0.5), 0.5)
        except KeyError:
            w_pref_cat = w_pref_sub = w_pref_ind = -1.0
            w_q = 0.5

        proj = self._project_idx.reindex(project_ids)
        p_cat = proj["project_category"].to_numpy(dtype=np.float32, copy=True)
        p_sub = proj["project_sub_category"].to_numpy(dtype=np.float32, copy=True)
        p_ind = proj["project_industry_code"].to_numpy(dtype=np.float32, copy=True)
        p_avg = proj["project_average_score"].to_numpy(dtype=np.float32, copy=True)
        for arr in (p_cat, p_sub, p_ind, p_avg):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        match_cat = ((p_cat == w_pref_cat) & (w_pref_cat != -1)).astype(np.float32)
        match_sub = ((p_sub == w_pref_sub) & (w_pref_sub != -1)).astype(np.float32)
        match_ind = ((p_ind == w_pref_ind) & (w_pref_ind != -1)).astype(np.float32)
        gap = np.abs(np.clip(w_q, 0.0, 1.0) - np.clip(p_avg / 5.0, 0.0, 1.0))

        return np.stack([match_cat, match_sub, match_ind, gap], axis=1)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_parquet(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required parquet file not found: {path}")
        return pd.read_parquet(path)

    def _fit_normalizers(self) -> None:
        """Compute z-score statistics from feature tables. Replaces zero std with 1."""
        w_arr = self.worker_features[list(WORKER_FEATURE_COLS)].to_numpy(dtype=np.float32)
        w_arr = np.nan_to_num(w_arr, nan=0.0)
        self._w_mean = w_arr.mean(axis=0)
        self._w_std = w_arr.std(axis=0)
        self._w_std[self._w_std == 0.0] = 1.0

        p_arr = self.project_features[list(PROJECT_FEATURE_COLS)].to_numpy(dtype=np.float32)
        p_arr = np.nan_to_num(p_arr, nan=0.0)
        self._p_mean = p_arr.mean(axis=0)
        self._p_std = p_arr.std(axis=0)
        self._p_std[self._p_std == 0.0] = 1.0


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def make_env(split: str = "train",
              processed_dir: str = "processed",
              reward_mode: RewardMode = "worker",
              candidate_mode: CandidateMode = "event_group",
              max_candidates: int = 20,
              seed: int = 42,
              normalize_features: bool = False) -> CrowdRecEnv:
    """Helper that wires standard processed paths into ``CrowdRecEnv``.

    Examples:
        >>> env = make_env("train", reward_mode="worker")
        >>> obs = env.reset()
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    cfg = EnvConfig(
        events_path=os.path.join(processed_dir, f"{split}_events.parquet"),
        worker_features_path=os.path.join(processed_dir, "worker_features.parquet"),
        project_features_path=os.path.join(processed_dir, "project_features.parquet"),
        reward_mode=reward_mode,
        candidate_mode=candidate_mode,
        max_candidates=max_candidates,
        seed=seed,
        normalize_features=normalize_features,
    )
    return CrowdRecEnv(cfg)


__all__ = [
    "CandidateMode",
    "CrowdRecEnv",
    "EnvConfig",
    "MATCH_FEATURE_COLS",
    "PROJECT_FEATURE_COLS",
    "WORKER_FEATURE_COLS",
    "make_env",
]
