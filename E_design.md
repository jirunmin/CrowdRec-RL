# E 部分交付：Baseline 策略与评估框架设计说明

> 对应 `分工.md` 中 E 角色的全部交付物：
>
> - `baselines/greedy_base.py` — Greedy 策略基类（统一接口）
> - `baselines/greedy_worker.py` — Greedy-Worker 策略（LR 学权重，worker reward 优化）
> - `baselines/greedy_requester.py` — Greedy-Requester 策略（LR 学权重，requester reward 优化）
> - `baselines/random_baseline.py` — Random 策略（均匀随机）
> - `baselines/dqn_policy.py` — DQN / Double-DQN 策略封装（接入 C/D 角色模型）
> - `baselines/find_weights_max_reward.py` — LR 权重学习脚本
> - `evaluation/evaluate.py` — 统一评估框架
> - `evaluation/metrics.py` — 评价指标计算
> - `visualization/plotting.py` — 可视化工具
> - 实验结果 JSON 文件 + 本设计文档

---

## 1. 总体定位

E 角色负责 **Baseline 实现、统一评估、实验分析与展示**。核心职责：

1. 实现 Random / Greedy / DQN / Double-DQN 四类方法的评估接口
2. 建立统一的评估框架，保证所有方法在相同条件下公平对比
3. 收集实验结果，生成可视化图表，撰写分析报告

### 设计原则

- **统一接口**：所有策略实现 `select_action(obs) → int` 接口，评估框架无需区分方法类型
- **公平对比**：所有方法使用相同的环境、数据集、随机种子、评估指标
- **可复现性**：固定 seed=42，结果完全可复现

---

## 2. Baseline 策略设计

### 2.1 Random Baseline

**文件**：`baselines/random_baseline.py`

最简单的基线方法，从有效候选中均匀随机选择。

```
π(a|s) = 1 / |valid_actions|
```

作用：
- **性能下界**：任何智能方法都应显著优于随机
- **验证环境**：如果 Random 异常，说明环境有问题

### 2.2 Greedy Baseline（LR 学权重）

**文件**：`baselines/greedy_base.py`、`greedy_worker.py`、`greedy_requester.py`

#### 核心设计

Greedy 策略使用全部 10 个可观测特征，线性加权打分：

```
score(s, a) = w^T · φ(s, a) = Σ w_i × feature_i
```

其中 φ(s, a) 包含以下 10 个特征：

| 特征         | 来源                | 含义                |
| ------------ | ------------------- | ------------------- |
| awards       | candidate_state[4]  | 项目总奖金          |
| match_cat    | candidate_state[10] | 类别匹配（0/1）     |
| match_sub    | candidate_state[11] | 子类匹配（0/1）     |
| match_ind    | candidate_state[12] | 行业匹配（0/1）     |
| quality_gap  | candidate_state[13] | 质量差距（0-1）     |
| featured     | candidate_state[6]  | 是否精选（0/1）     |
| avg_score    | candidate_state[7]  | 项目平均评分        |
| has_winner   | candidate_state[9]  | 是否有中标者（0/1） |
| worker_count | candidate_state[8]  | 参与 worker 数      |
| entry_count  | candidate_state[3]  | 已有提交数          |

#### 权重学习

传统 Greedy 的权重由人工设定（如 awards=2.0, match=1.5），缺乏数据支撑。本项目使用 **线性回归（LR）从训练集数据中学习最优权重**。

**方法**：
1. 从 `train_events.parquet` 加载全部事件
2. 构造 10 维特征矩阵 X 和 reward 向量 y
3. 训练线性回归：`y = w^T · X + bias`
4. 系数 w 即为最优权重

**优势**：
- 权重由数据决定，不依赖人工直觉
- 训练速度极快（解析解，几秒完成）
- 可解释性强（每个权重对应一个特征的重要性）

**局限**：
- 线性模型，无法捕捉特征间的交互效应
- 只能优化单步 reward，不考虑长期累积

**脚本**：`baselines/find_weights_max_reward.py`，运行后输出可直接复制到代码中的权重。

#### Greedy-Worker vs Greedy-Requester

两个策略使用**相同的 10 个特征**，区别仅在于 LR 训练时使用的 reward 不同：

| 策略             | LR 训练目标      | 优化方向              |
| ---------------- | ---------------- | --------------------- |
| Greedy-Worker    | worker_reward    | 最大化 worker 利益    |
| Greedy-Requester | requester_reward | 最大化 requester 利益 |

### 2.3 DQN / Double-DQN 策略封装

**文件**：`baselines/dqn_policy.py`

将 C 角色的 DQN 和 D 角色的 Double-DQN 模型封装为标准策略接口，接入评估框架。

```python
class DQNPolicy:
    def select_action(self, obs) -> int:
        state = flatten(obs)  # 292 维
        return agent.select_action(state, valid_mask=obs["valid_mask"], eval_mode=True)
```

**关键设计**：
- `agent_type="dqn"` 加载 C 的模型，`agent_type="double-dqn"` 加载 D 的模型
- 根据 `reward_mode` 自动加载对应的模型文件（worker / requester）
- `eval_mode=True`：关闭 epsilon-greedy 探索，纯贪心决策
- 支持 GPU 加速（`device="auto"` 自动检测）

---

## 3. 评估框架设计

### 3.1 统一评估接口

**文件**：`evaluation/evaluate.py`

核心函数 `evaluate_policy(env_factory, policy_fn)`：

```
对每个 episode：
    obs = env.reset()
    while not done:
        action = policy_fn(obs)           ← 策略选动作
        obs, reward, done, info = env.step(action)  ← 环境执行
        记录 reward, hit, step
返回 {total_reward, avg_reward, hit_rate, ...}
```

**设计要点**：
- `env_factory`：无参工厂函数，每次调用创建新环境实例
- `policy_fn`：接受 obs 返回 action 的函数，与策略类型无关
- `seed`：固定随机种子，保证可复现性

### 3.2 命令行接口

```bash
python -m evaluation.evaluate \
    --split test \              # train | val | test
    --mode worker \             # worker | requester
    --candidate_mode event_group \  # event_group | top_k
    --data_dir processed \      # 数据目录
    --methods random greedy-worker greedy-requester dqn \  # 指定方法
    --output results.json \     # 输出路径
    --quiet                     # 关闭详细输出
```

**灵活组合**：
- `--methods`：只跑指定方法，跳过其他（节省时间）
- `--data_dir`：支持不同数据集（如 1:19 的 processed_1to20）
- 不加 `--methods`：跑全部方法（包括 DQN）

### 3.3 评价指标

| 指标           | 公式              | 含义                        |
| -------------- | ----------------- | --------------------------- |
| **Avg Reward** | Σ reward / steps  | 平均每步 reward（核心指标） |
| **Hit@1**      | 命中次数 / 总步数 | 推荐命中率                  |
| **NDCG@1**     | DCG / IDCG        | 命中时的 reward 归一化      |
| **MRR**        | Σ 1/rank / Q      | 正确答案排名的倒数          |

> 注：每步只推荐 1 个候选（K=1），Hit@5 / NDCG@5 无意义，不输出。

---

## 4. 与其他角色的对接

| 角色 | 对接点                                   | E 的用法                   |
| ---- | ---------------------------------------- | -------------------------- |
| A    | `processed/*.parquet`                    | 直接读取预处理后的数据     |
| B    | `src/env.py` 的 `make_env()`             | 创建环境，获取 obs         |
| B    | `info["hit"]`                            | 每步返回 hit，计算 Hit@1   |
| B    | `info["ground_truth_index"]`             | 真实选择索引，用于指标计算 |
| C    | `c_basic_dqn/agent_dqn.py` 的 `DQNAgent` | 加载训练好的 DQN 模型      |
| C    | `c_basic_dqn/basic_dqn_best_*.pth`       | 模型权重文件               |
| D    | `d_double_dqn/agent_ddqn.py`             | 加载训练好的 Double-DQN 模型 |
| D    | `d_double_dqn/double_dqn_best_*.pth`     | Double-DQN 模型权重文件    |

### 接口规范

C 角色的 DQN 模型需提供：
- `select_action(state, valid_mask, eval_mode)` → int
- `load_model(path)` / `save_model(path)`
- 输入：292 维展平状态向量
- 输出：0-19 的动作索引

E 角色通过 `DQNPolicy` 封装类自动适配，无需修改 C 的代码。

---

## 5. 文件清单

```
CrowdRec-RL/
├── baselines/
│   ├── __init__.py              # 导出所有策略类
│   ├── greedy_base.py           # Greedy 策略基类（10 特征线性加权）
│   ├── greedy_worker.py         # Greedy-Worker（worker reward 学权重）
│   ├── greedy_requester.py      # Greedy-Requester（requester reward 学权重）
│   ├── random_baseline.py       # Random 策略
│   ├── dqn_policy.py            # DQN 策略封装
│   └── find_weights_max_reward.py  # LR 权重学习脚本
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py              # 统一评估框架
│   └── metrics.py               # 评价指标计算
├── visualization/
│   ├── __init__.py
│   └── plotting.py              # 可视化工具
├── experiments/
│   └── results/                 # 实验结果 JSON
│       ├── random_greedy_test_worker.json       # Random + Greedy (1:2)
│       ├── random_greedy_test_requester.json
│       ├── random_greedy_test2_worker.json      # Random + Greedy (1:19)
│       ├── random_greedy_test2_requester.json
│       ├── dqn_test_worker.json                 # DQN (1:2)
│       ├── dqn_test_requester.json
│       ├── dqn_test1_worker.json                # DQN (1:9)
│       ├── dqn_test1_requester.json
│       ├── dqn_test2_worker.json                # DQN (1:19)
│       ├── dqn_test2_requester.json
│       ├── ddqn_test_worker.json                # Double-DQN (1:2)
│       ├── ddqn_test_requester.json
│       ├── ddqn_test1_worker.json               # Double-DQN (1:9)
│       ├── ddqn_test1_requester.json
│       ├── ddqn_test2_worker.json               # Double-DQN (1:19)
│       └── ddqn_test2_requester.json
└── E_design.md                  # 本文档
```

---

## 6. 实验结果摘要

### Test Set（1:2，3 候选，68,512 步）

| 方法             | Worker Avg Reward | Worker Hit@1 | Requester Avg Reward | Requester Hit@1 |
| ---------------- | ----------------- | ------------ | -------------------- | --------------- |
| Random           | 0.498             | 33.2%        | 0.604                | 33.2%           |
| Greedy-Worker    | 0.835             | 56.1%        | 1.015                | 56.1%           |
| Greedy-Requester | 0.829             | 55.7%        | 1.009                | 55.7%           |
| **DQN**          | **0.958**         | **64.6%**    | **1.170**            | **64.8%**       |
| Double-DQN       | 0.874             | 59.0%        | 1.129                | 62.7%           |

### Test1 Set（1:9，~10 候选，64,408 步）

| 方法         | Worker Avg Reward | Worker Hit@1 | Requester Avg Reward | Requester Hit@1 |
| ------------ | ----------------- | ------------ | -------------------- | --------------- |
| DQN          | 0.268             | 18.1%        | 0.334                | 18.5%           |
| Double-DQN   | 0.183             | 12.3%        | 0.208                | 11.5%           |

### Test2 Set（1:19，~20 候选，62,781 步）

| 方法             | Worker Avg Reward | Worker Hit@1 | Requester Avg Reward | Requester Hit@1 |
| ---------------- | ----------------- | ------------ | -------------------- | --------------- |
| Random           | 0.079             | 5.3%         | 0.095                | 5.3%            |
| Greedy-Worker    | 0.274             | 18.6%        | 0.334                | 18.6%           |
| Greedy-Requester | 0.264             | 17.8%        | 0.322                | 17.8%           |
| DQN              | 0.134             | 9.0%         | 0.168                | 9.3%            |
| Double-DQN       | 0.091             | 6.1%         | 0.106                | 5.9%            |

> 注：Test1 集无 baseline 结果；DQN/DDQN 在 1:2 数据集上训练，测试泛化到 1:9 和 1:19。
> DQN/DDQN 在 test2 上表现低于 Greedy，说明 RL 模型在训练分布外泛化能力有限。

详细分析见 `E_res.md`。
