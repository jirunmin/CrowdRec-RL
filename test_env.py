"""Smoke tests for CrowdRecEnv.

Run end-to-end with a random policy on a tiny slice of train data, then on the
real val/test splits to sanity-check shapes, masking, and reward sums.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
TMP = ROOT / "tmp_test"
TMP.mkdir(exist_ok=True)

from src.env import CrowdRecEnv, EnvConfig, make_env  # noqa: E402
from src.reward import (  # noqa: E402
    compute_reward_array,
    pick_precomputed_column,
)


def make_mini_split() -> EnvConfig:
    """Take the first 10k rows of train and persist as a tiny split."""
    train = pd.read_parquet(PROCESSED / "train_events.parquet")
    mini = train.iloc[:10_000].copy()
    mini_path = TMP / "mini_events.parquet"
    mini.to_parquet(mini_path, index=False)
    return EnvConfig(
        events_path=str(mini_path),
        worker_features_path=str(PROCESSED / "worker_features.parquet"),
        project_features_path=str(PROCESSED / "project_features.parquet"),
        candidates_path=str(PROCESSED / "candidates.parquet"),
        reward_mode="worker",
        candidate_mode="event_group",
        max_candidates=20,
        seed=0,
    )


def test_mini_event_group() -> None:
    print("[1] mini event_group worker reward")
    cfg = make_mini_split()
    env = CrowdRecEnv(cfg)
    obs = env.reset()

    # Shape checks.
    assert obs["worker_state"].shape == (env.worker_dim,)
    assert obs["candidate_state"].shape == (env.max_candidates, env.candidate_dim)
    assert obs["valid_mask"].shape == (env.max_candidates,)
    assert obs["valid_mask"].any(), "first step should have at least one valid candidate"

    # Roll out the entire mini split with random policy.
    rng = np.random.default_rng(123)
    total_random = 0.0
    total_oracle = 0.0
    n_steps = len(env)
    n_hits_random = 0

    obs = env.reset()
    for _ in range(n_steps):
        valid = np.flatnonzero(obs["valid_mask"])
        action = int(rng.choice(valid))
        obs_next, reward, done, info = env.step(action)
        total_random += reward
        n_hits_random += info["hit"] if info["ground_truth_index"] >= 0 else 0
        # Oracle baseline: always pick the ground-truth.
        gt = info["ground_truth_index"]
        if gt >= 0:
            # We need the precomputed reward of the gt – pull from the step we just left.
            step = env._steps[env._cursor - 1]  # noqa: SLF001 (test-only)
            total_oracle += float(step["candidate_rewards"][gt])
        obs = obs_next

    assert done is True

    # Sanity: oracle reward should be >= random reward.
    print(f"   steps={n_steps}, random_total={total_random:.2f}, oracle_total={total_oracle:.2f}, "
          f"random_hit_rate={n_hits_random / max(n_steps, 1):.3f}")
    assert total_oracle >= total_random - 1e-6


def test_mini_requester_reward() -> None:
    print("[2] mini event_group requester reward")
    cfg = make_mini_split()
    cfg = EnvConfig(**{**cfg.__dict__, "reward_mode": "requester"})
    env = CrowdRecEnv(cfg)
    obs = env.reset()

    rng = np.random.default_rng(7)
    total = 0.0
    while True:
        valid = np.flatnonzero(obs["valid_mask"])
        action = int(rng.choice(valid))
        obs, r, done, info = env.step(action)
        total += r
        if done:
            break
    print(f"   total requester reward (random): {total:.2f}")


def test_mini_top_k() -> None:
    print("[3] mini top_k worker reward")
    cfg = make_mini_split()
    cfg = EnvConfig(**{**cfg.__dict__, "candidate_mode": "top_k"})
    env = CrowdRecEnv(cfg)

    obs = env.reset()
    n_steps = len(env)
    rng = np.random.default_rng(7)
    total = 0.0
    n_hits = 0
    n_with_gt = 0
    for _ in range(n_steps):
        valid = np.flatnonzero(obs["valid_mask"])
        action = int(rng.choice(valid))
        obs, r, done, info = env.step(action)
        total += r
        if info["ground_truth_index"] >= 0:
            n_with_gt += 1
            if info["hit"]:
                n_hits += 1
    print(f"   top_k steps={n_steps}, gt-resolvable={n_with_gt}, "
          f"random_hit_rate={n_hits / max(n_with_gt, 1):.3f}, total_reward={total:.2f}")


def test_reward_consistency_with_event_group() -> None:
    """Compute env-style reward for a random rollout via reward.py and check it matches."""
    print("[4] reward.py vs precomputed parquet match")
    df = pd.read_parquet(PROCESSED / "train_events.parquet").iloc[:50_000]
    for mode in ("worker", "requester"):
        col = pick_precomputed_column(mode)
        recomputed = compute_reward_array(df, mode)
        precomputed = df[col].to_numpy(dtype=np.float32)
        diff = np.abs(recomputed - precomputed).max()
        print(f"   mode={mode}: max abs diff = {diff:.4f}")
        # event_stream uses the same coefficients, so we expect very small diff.
        assert diff < 1e-3, f"reward.py disagrees with parquet for mode={mode}"


def test_make_env_val_small() -> None:
    """Build env on the real val split, but only roll out a few hundred steps."""
    print("[5] make_env('val') – partial rollout")
    t0 = time.time()
    env = make_env("val", processed_dir=str(PROCESSED), reward_mode="worker")
    build_t = time.time() - t0
    print(f"   build steps: {len(env)} in {build_t:.1f}s")

    rng = np.random.default_rng(0)
    total = 0.0
    n_hits = 0
    obs = env.reset()
    for i in range(min(500, len(env))):
        valid = np.flatnonzero(obs["valid_mask"])
        action = int(rng.choice(valid))
        obs, r, done, info = env.step(action)
        total += r
        if info["hit"]:
            n_hits += 1
        if done:
            break
    print(f"   first 500 steps: total_reward={total:.2f}, hits={n_hits}")


def test_no_gt_index_leak() -> None:
    """Regression: gt_index must not be biased toward 0.

    If candidates are sorted by `label` descending (an earlier bug), the
    positive sample sits at index 0 in nearly every group and a trivial
    'always pick action 0' policy reaches 100% hit rate.
    """
    print("[6] gt_index distribution is unbiased + always-0 baseline ≈ 1/3")
    env = make_env("val", processed_dir=str(PROCESSED), reward_mode="worker")
    from collections import Counter
    dist = Counter()
    for s in env._steps:
        if s["gt_index"] >= 0:
            dist[s["gt_index"]] += 1
    total = sum(dist.values())
    pct0 = dist[0] / total
    print(f"   gt_index=0 share: {pct0:.3f} (expect ≈ 1/3)")
    assert 0.25 < pct0 < 0.45, f"gt_index leak detected: {pct0:.3f}"

    # Always-0 baseline check.
    obs = env.reset()
    hits, resolvable = 0, 0
    for _ in range(min(2000, len(env))):
        obs, _, done, info = env.step(0)
        if info["ground_truth_index"] >= 0:
            resolvable += 1
            if info["hit"]:
                hits += 1
        if done:
            break
    rate = hits / max(resolvable, 1)
    print(f"   always-0 hit rate: {rate:.3f} (must NOT be ≈1.0)")
    assert rate < 0.45, f"label leak: always-0 policy gets {rate:.3f}"


def test_match_features_match_parquet() -> None:
    """Regression: env._match_block must reproduce parquet match_* columns.

    Two earlier bugs:
        - falsy-0 in `pref_industry`: ``0.0 or -1`` → -1 wrongly nukes match_industry
        - worker_quality not falling back to worker_quality_pred for unlabeled
          workers, breaking match_quality_gap
    """
    print("[7] env._match_block consistent with parquet match_* columns")
    env = make_env("train", processed_dir=str(PROCESSED), reward_mode="worker")
    sample = env.events_df.head(5_000)
    mismatches = {"cat": 0, "sub": 0, "ind": 0, "gap": 0}
    for _, ev in sample.iterrows():
        m = env._match_block(int(ev["worker"]),
                              np.array([int(ev["project_id"])], dtype=np.int64))
        if abs(m[0, 0] - ev["match_category"]) > 1e-5:
            mismatches["cat"] += 1
        if abs(m[0, 1] - ev["match_sub_category"]) > 1e-5:
            mismatches["sub"] += 1
        if abs(m[0, 2] - ev["match_industry"]) > 1e-5:
            mismatches["ind"] += 1
        if abs(m[0, 3] - ev["match_quality_gap"]) > 1e-5:
            mismatches["gap"] += 1
    print(f"   mismatches over 5000 events: {mismatches}")
    for k, v in mismatches.items():
        assert v == 0, f"match_{k} drift: {v} mismatches"


def main() -> None:
    test_reward_consistency_with_event_group()
    test_mini_event_group()
    test_mini_requester_reward()
    test_mini_top_k()
    test_make_env_val_small()
    test_no_gt_index_leak()
    test_match_features_match_parquet()
    print("\nAll env smoke tests passed.")


if __name__ == "__main__":
    main()
