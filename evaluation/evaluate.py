"""
Unified Evaluation Framework for Crowd Recommendation

Author: E角色
Description: 统一的评估接口，用于公平对比所有方法（Baseline + DQN系列）
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime

from .metrics import compute_all_metrics


PolicyFn = Callable[[Dict[str, Any]], int]
"""
策略函数类型定义

输入：环境观测字典 obs（包含worker_state, candidate_state, valid_mask, info）
输出：动作索引 action（整数，0到K-1）
"""


def evaluate_policy(
    env_factory: Callable,
    policy_fn: PolicyFn,
    n_episodes: int = 1,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, float]:
    """
    统一评估接口 - 这是整个项目的核心评估函数！

    设计理念：
    ==========
    1. **通用性**：任何策略（Random、Greedy、DQN等）都可以用这个函数评估
    2. **公平性**：所有方法使用相同的环境、相同的随机种子、相同的指标
    3. **可复现性**：通过seed参数确保结果可复现

    工作流程：
    ==========
    对于每个episode：
        a. 通过env_factory()创建新的环境实例
        b. 调用env.reset(seed)初始化环境，获得初始观测obs
        c. 循环：
           - 调用policy_fn(obs)获取动作action
           - 调用env.step(action)执行动作，获得(next_obs, reward, done, info)
           - 累积reward、统计hits、记录步数
           - 更新obs = next_obs
           - 如果done=True，结束当前episode
        d. 记录该episode的结果

    最终计算：
    - 总奖励、平均奖励、奖励标准差
    - Hit@1命中率
    - 其他统计信息

    Args:
        env_factory: 无参工厂函数，返回配置好的环境实例
            示例：lambda: make_env(split="val", reward_mode="worker")
        policy_fn: 策略函数，输入obs字典，输出action整数
            示例：RandomPolicy().select_action
        n_episodes: 运行多少个完整的episode（默认1）
            通常验证集和测试集各跑1次就够了
        seed: 随机种子（默认42）
            用于环境初始化，保证可复现性
        verbose: 是否打印详细的进度信息（默认True）

    Returns:
        Dict[str, float]: 包含以下键的字典：
            - total_reward: 所有episode的总累积奖励
            - avg_reward: 平均每步奖励
            - hit_rate: Hit@1命中率
            - std_reward: 奖励标准差（稳定性指标）
            - min_reward: 单个episode最小总奖励
            - max_reward: 单个episode最大总奖励
            - total_steps: 总步数
            - n_episodes: 运行的episode数

    使用示例：
    ========
    >>> from baselines import RandomPolicy
    >>> from src.env import make_env
    >>>
    >>> # 创建环境和策略
    >>> env_factory = lambda: make_env(split="val", reward_mode="worker")
    >>> policy = RandomPolicy(seed=42)
    >>>
    >>> # 运行评估
    >>> results = evaluate_policy(env_factory, policy.select_action)
    >>> print(f"Hit Rate: {results['hit_rate']:.4f}")
    """
    all_rewards = []
    all_hits = []
    all_steps = []
    all_predictions = []
    all_ground_truth = []
    all_step_rewards = []

    total_reward = 0.0
    total_hits = 0
    total_steps = 0

    for ep in range(n_episodes):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running Episode {ep+1}/{n_episodes}")
            print(f"{'='*70}")

        ep_seed = seed + ep if seed is not None else None

        try:
            env = env_factory()

            if hasattr(env, 'reset'):
                reset_params = {}
                if 'seed' in env.reset.__code__.co_varnames:
                    reset_params['seed'] = ep_seed
                obs = env.reset(**reset_params)
            else:
                raise AttributeError("Environment has no reset method")

        except Exception as e:
            print(f"❌ Error creating/resetting environment: {e}")
            continue

        done = False
        ep_reward = 0.0
        ep_hits = 0
        ep_steps = 0

        while not done:
            try:
                action = policy_fn(obs)

                obs_next, reward, done, info = env.step(action)

                ep_reward += reward
                ep_hits += info.get("hit", 0)
                ep_steps += 1

                # 收集每步数据用于计算详细指标
                all_predictions.append(int(action))
                all_ground_truth.append(info.get("ground_truth_index", -1))
                all_step_rewards.append(float(reward))

                if verbose and ep_steps % 20000 == 0:
                    print(f"  Step {ep_steps}: "
                          f"reward={ep_reward:.2f}, "
                          f"hits={ep_hits}, "
                          f"hit_rate={ep_hits/max(ep_steps,1):.4f}")

                obs = obs_next

            except Exception as e:
                print(f"❌ Error at step {ep_steps}: {e}")
                break

        all_rewards.append(ep_reward)
        all_hits.append(ep_hits)
        all_steps.append(ep_steps)

        total_reward += ep_reward
        total_hits += ep_hits
        total_steps += ep_steps

        if verbose:
            print(f"\n✓ Episode {ep+1} completed:")
            print(f"  Steps: {ep_steps}")
            print(f"  Reward: {ep_reward:.2f}")
            print(f"  Hits: {ep_hits} (rate: {ep_hits/max(ep_steps,1):.4f})")

    results = {
        "total_reward": float(total_reward),
        "avg_reward": float(total_reward / max(total_steps, 1)),
        "hit_rate": float(total_hits / max(total_steps, 1)),
        "std_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
        "min_reward": float(min(all_rewards)) if all_rewards else 0.0,
        "max_reward": float(max(all_rewards)) if all_rewards else 0.0,
        "total_steps": total_steps,
        "n_episodes": n_episodes,
    }

    # 计算详细指标（NDCG, MRR 等）
    if all_predictions:
        detailed = compute_all_metrics(all_predictions, all_ground_truth, all_step_rewards)
        results.update(detailed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluation Results Summary ({n_episodes} episode(s))")
        print(f"{'='*70}")
        print(f"Total Steps:     {total_steps:>10,}")
        print(f"Total Reward:    {results['total_reward']:>10.2f}")
        print(f"Avg Reward:      {results['avg_reward']:>10.4f}")
        print(f"Hit@1 Rate:      {results['hit_rate']:>10.4f}")
        print(f"NDCG@1:          {results.get('ndcg_1', 0):>10.4f}")
        print(f"MRR:             {results.get('mrr', 0):>10.4f}")
        print(f"Reward Std:      {results['std_reward']:>10.4f}")
        if n_episodes > 1:
            print(f"Per-Episode Rewards: {[f'{r:.2f}' for r in all_rewards]}")
        print(f"{'='*70}\n")

    return results


def run_full_evaluation(
    split: str = "val",
    reward_mode: str = "worker",
    candidate_mode: str = "event_group",
    processed_dir: str = "processed",
    output_path: Optional[str] = None,
    verbose: bool = True,
    methods: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    一键运行所有baseline方法的完整评估

    这是E角色最常用的函数！它会：
    1. 创建指定split和reward_mode的环境
    2. 实例化所有3个baseline策略（Random, Greedy-Worker, Greedy-Requester）
    3. 对每个策略调用evaluate_policy进行评估
    4. 收集所有结果并返回列表
    5. 可选：保存为JSON文件供后续分析

    Args:
        split: 数据集划分 ("train" | "val" | "test")
        reward_mode: 奖励模式 ("worker" | "requester")
        output_path: 结果保存路径（可选，如"experiments/results/val_worker.json"）
        verbose: 是否打印详细信息

    Returns:
        List[Dict]: 每个元素是一个方法的结果字典，包含：
            - method_name: 方法名称
            - results: evaluate_policy返回的完整结果字典
            - timestamp: 评估时间戳

    使用示例：
    ========
    >>> # 在验证集上评估Worker模式的所有baseline
    >>> results = run_full_evaluation(split="val", reward_mode="worker")
    >>>
    >>> # 查看结果
    >>> for r in results:
    ...     print(f"{r['method_name']}: Hit@1={r['results']['hit_rate']:.4f}")
    """
    try:
        from baselines.random_baseline import RandomPolicy
        from baselines.greedy_worker import GreedyWorkerPolicy
        from baselines.greedy_requester import GreedyRequesterPolicy
    except ImportError as e:
        print(f"❌ Cannot import baseline policies: {e}")
        print("   Make sure you're running from the project root directory (CrowdRec-RL/)")
        return []

    # 尝试导入 DQN
    try:
        from baselines.dqn_policy import DQNPolicy
        has_dqn = True
    except ImportError:
        has_dqn = False

    try:
        from src.env import make_env
        # from src.fastenv import make_fast_env
    except ImportError:
        print("❌ Cannot import make_env from src.env. Make sure you're running from project root.")
        return []

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Full Evaluation Pipeline")
        print(f"# Split: {split.upper()} | Reward Mode: {reward_mode.upper()} | Candidate Mode: {candidate_mode.upper()}")
        print(f"{'#'*70}\n")

    env_factory = lambda: make_env(split=split, reward_mode=reward_mode, candidate_mode=candidate_mode, processed_dir=processed_dir)
    # env_factory = lambda: make_fast_env(split=split, reward_mode=reward_mode, candidate_mode=candidate_mode, processed_dir=processed_dir)
    
    all_policies = [
        ("Random", RandomPolicy(seed=42)),
        ("Greedy-Worker", GreedyWorkerPolicy(seed=42)),
        ("Greedy-Requester", GreedyRequesterPolicy(seed=42)),
    ]

    # 判断需要加载哪些 RL 模型
    methods_lower = [m.lower() for m in methods] if methods else []
    need_dqn = methods is None or "dqn" in methods_lower
    need_double_dqn = methods is None or "double-dqn" in methods_lower

    # 根据 reward_mode 添加对应的 DQN 模型
    if has_dqn and need_dqn:
        if reward_mode == "worker":
            try:
                dqn = DQNPolicy(model_path="c_basic_dqn/basic_dqn_best_worker_model.pth", device="auto")
                all_policies.append(("DQN", dqn))
            except Exception as e:
                print(f"⚠️ 无法加载 DQN worker 模型: {e}")
        elif reward_mode == "requester":
            try:
                dqn = DQNPolicy(model_path="c_basic_dqn/basic_dqn_best_requester_model.pth", device="auto")
                all_policies.append(("DQN", dqn))
            except Exception as e:
                print(f"⚠️ 无法加载 DQN requester 模型: {e}")

    # 根据 reward_mode 添加对应的 Double DQN 模型
    if has_dqn and need_double_dqn:
        if reward_mode == "worker":
            try:
                ddqn = DQNPolicy(model_path="d_double_dqn/double_dqn_best_worker_model.pth", device="auto", agent_type="double-dqn")
                all_policies.append(("Double-DQN", ddqn))
            except Exception as e:
                print(f"⚠️ 无法加载 Double-DQN worker 模型: {e}")
        elif reward_mode == "requester":
            try:
                ddqn = DQNPolicy(model_path="d_double_dqn/double_dqn_best_requester_model.pth", device="auto", agent_type="double-dqn")
                all_policies.append(("Double-DQN", ddqn))
            except Exception as e:
                print(f"⚠️ 无法加载 Double-DQN requester 模型: {e}")

    if methods is not None:
        methods_lower = [m.lower() for m in methods]
        policies = [(n, p) for n, p in all_policies if n.lower() in methods_lower]
    else:
        policies = all_policies

    all_results = []

    for name, policy in policies:
        if verbose:
            print(f"\n{'→'*30}")
            print(f"→ Evaluating: {name}")
            print(f"{'→'*30}\n")

        start_time = datetime.now()

        try:
            results = evaluate_policy(
                env_factory=env_factory,
                policy_fn=policy.select_action,
                n_episodes=1,
                seed=42,
                verbose=verbose
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            result_entry = {
                "method_name": name,
                "split": split,
                "reward_mode": reward_mode,
                "results": results,
                "evaluation_time_seconds": elapsed,
                "timestamp": datetime.now().isoformat()
            }

            all_results.append(result_entry)

            if verbose:
                print(f"\n✓ {name} evaluation completed in {elapsed:.2f}s")
                print(f"  Total Reward: {results['total_reward']:.2f}")
                print(f"  Avg Reward: {results['avg_reward']:.4f}")
                print(f"  Hit@1 Rate: {results['hit_rate']:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

            all_results.append({
                "method_name": name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    if output_path and all_results:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

        if verbose:
            print(f"\n💾 Results saved to: {output_file.absolute()}")

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Evaluation Complete!")
        print(f"# Methods evaluated: {len([r for r in all_results if 'error' not in r])}")
        print(f"{'#'*70}\n")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--mode", type=str, default="worker",
                        choices=["worker", "requester"],
                        help="Reward mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Specific methods to run (e.g. --methods greedy-worker greedy-requester)")
    parser.add_argument("--candidate_mode", type=str, default="event_group",
                        choices=["event_group", "top_k"],
                        help="Candidate mode: event_group (3 candidates) or top_k (20 candidates)")
    parser.add_argument("--data_dir", type=str, default="processed",
                        help="Processed data directory (default: processed)")

    args = parser.parse_args()

    print("="*70)
    print("Running Baseline Evaluations")
    print(f"Split: {args.split} | Mode: {args.mode}")
    print("="*70)

    results = run_full_evaluation(
        split=args.split,
        reward_mode=args.mode,
        candidate_mode=args.candidate_mode,
        processed_dir=args.data_dir,
        output_path=args.output,
        verbose=not args.quiet,
        methods=args.methods
    )

    print("\nSummary Table:")
    print("-"*70)
    print(f"{'Method':<20} {'Total Reward':>15} {'Avg Reward':>12} {'Hit@1':>10}")
    print("-"*70)

    for r in results:
        if 'results' in r:
            res = r['results']
            print(f"{r['method_name']:<20} "
                  f"{res['total_reward']:>15.2f} "
                  f"{res['avg_reward']:>12.4f} "
                  f"{res['hit_rate']:>10.4f}")
        else:
            print(f"{r['method_name']:<20} {'ERROR':>37}")

    print("-"*70)
