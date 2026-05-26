"""
Reward functions for CrowdRec-RL.

Two reward formulations:
- worker     : maximizes participant interest. Pays for participation, biased to
               quality, with extra credit for winning / being a finalist.
- requester  : maximizes platform/requester benefit. Cares about high-quality
               participation that produces a winner.

Both functions return 0 for negative samples (i.e. workers who saw the project
but did not engage with it). Reward is computed only when the env *recommends*
the project; if the policy picks a different project, the env applies the
"counterfactual" rule defined in env.py (typically reward = 0 for non-chosen).

These coefficients match the offline pre-computation in
``src/event_stream.py`` so that the env can either recompute on the fly or
trust the precomputed ``reward_worker`` / ``reward_requester`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RewardMode = Literal["worker", "requester"]


@dataclass(frozen=True)
class WorkerRewardCoef:
    """Coefficients for the worker (participant) reward."""

    base: float = 1.0           # 参与基础分
    quality: float = 0.5        # 质量奖励
    winner: float = 2.0         # 中标奖励
    finalist: float = 1.0       # 入围奖励


@dataclass(frozen=True)
class RequesterRewardCoef:
    """Coefficients for the requester reward."""

    quality: float = 2.0        # 质量贡献
    winner: float = 1.0         # 选出优胜者


WORKER_COEF = WorkerRewardCoef()
REQUESTER_COEF = RequesterRewardCoef()


def worker_reward(label: float | int | np.ndarray,
                  worker_quality: float | np.ndarray,
                  winner: float | int | np.ndarray,
                  finalist: float | int | np.ndarray,
                  coef: WorkerRewardCoef = WORKER_COEF) -> np.ndarray:
    """Worker-side reward.

    reward = label * (base + quality*q + winner_coef*winner + finalist_coef*finalist)

    Args:
        label:           1 if worker actually participated, 0 otherwise.
        worker_quality:  worker quality in [0, 1].
        winner:          0/1 indicator (cast from bool if needed).
        finalist:        0/1 indicator.
        coef:            Reward coefficients.

    Returns:
        np.ndarray (or scalar) of rewards. Returns 0 for label==0 cases.
    """
    label = np.asarray(label, dtype=np.float32)
    q = np.asarray(worker_quality, dtype=np.float32)
    w = np.asarray(winner, dtype=np.float32)
    f = np.asarray(finalist, dtype=np.float32)

    return label * (coef.base + coef.quality * q + coef.winner * w + coef.finalist * f)


def requester_reward(label: float | int | np.ndarray,
                     worker_quality: float | np.ndarray,
                     winner: float | int | np.ndarray,
                     coef: RequesterRewardCoef = REQUESTER_COEF) -> np.ndarray:
    """Requester-side reward.

    reward = label * (quality_coef * q + winner_coef * winner)
    """
    label = np.asarray(label, dtype=np.float32)
    q = np.asarray(worker_quality, dtype=np.float32)
    w = np.asarray(winner, dtype=np.float32)

    return label * (coef.quality * q + coef.winner * w)


def compute_reward_row(row: pd.Series, mode: RewardMode) -> float:
    """Compute reward for a single event row using ``label/winner/finalist/worker_quality``."""
    if mode == "worker":
        return float(worker_reward(
            row["label"], row["worker_quality"], row["winner"], row["finalist"],
        ))
    if mode == "requester":
        return float(requester_reward(
            row["label"], row["worker_quality"], row["winner"],
        ))
    raise ValueError(f"Unknown reward mode: {mode!r}")


def compute_reward_array(df: pd.DataFrame, mode: RewardMode) -> np.ndarray:
    """Vectorized reward over a DataFrame containing the required columns."""
    if mode == "worker":
        return worker_reward(
            df["label"].to_numpy(),
            df["worker_quality"].to_numpy(),
            df["winner"].to_numpy(),
            df["finalist"].to_numpy(),
        )
    if mode == "requester":
        return requester_reward(
            df["label"].to_numpy(),
            df["worker_quality"].to_numpy(),
            df["winner"].to_numpy(),
        )
    raise ValueError(f"Unknown reward mode: {mode!r}")


def pick_precomputed_column(mode: RewardMode) -> str:
    """Map reward mode -> column name in the preprocessed parquet."""
    return "reward_worker" if mode == "worker" else "reward_requester"


__all__ = [
    "RewardMode",
    "WorkerRewardCoef",
    "RequesterRewardCoef",
    "WORKER_COEF",
    "REQUESTER_COEF",
    "worker_reward",
    "requester_reward",
    "compute_reward_row",
    "compute_reward_array",
    "pick_precomputed_column",
]
