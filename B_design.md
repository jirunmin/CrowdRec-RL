# B 部分交付：RL 环境与 Reward 设计说明

> 对应 `task_division.md` 中 B 角色的全部交付物：
>
> - `src/env.py`
> - `src/reward.py`
> - 状态 / 动作 / 奖励设计说明（本文档）
> - 已通过 Random 策略的端到端 smoke test (`test_env.py`)

---

## 1. 总体定位

CrowdRec-RL 的训练数据是一条**严格按时序排列的离线事件流**，每条事件描述
"worker 在 timestamp 是否选择了某个 project"。环境的职责是：

1. 把事件流回放给 DQN-style agent，每一步给出当前 worker 与候选 project 集合；
2. agent 选一个候选项目（action），环境按 reward 函数返回标量奖励；
3. 前进到下一条事件，直到事件流耗尽 → `done = True`。

环境是**反事实 (counterfactual) 的**——它不模拟 worker 行为，只在 agent 命中
真实参与的项目时给奖励，否则给 0。这种"反事实回放"正是离线推荐 RL 的标准做法，
也保证了和 A 角色已经构造好的 `(reward_worker, reward_requester)` 列完全对齐。

---

## 2. 数据接口

环境直接消费 A 阶段产出的 parquet：

| 文件 | 用途 |
| --- | --- |
| `processed/train_events.parquet` / `val_events.parquet` / `test_events.parquet` | 时序划分后的 (worker, project, timestamp, label, reward_*) 事件 |
| `processed/worker_features.parquet` | Worker 静态特征查找表 |
| `processed/project_features.parquet` | Project 静态特征查找表 |
| `processed/candidates.parquet` | 每个正样本时刻的 Top-20 活跃项目（仅 `top_k` 模式需要） |

构造方式（推荐）：

```python
from src.env import make_env

env = make_env(
    split="train",            # train | val | test
    reward_mode="worker",     # worker | requester
    candidate_mode="event_group",  # event_group | top_k
)
obs = env.reset()
```

---

## 3. State（状态）设计

每一步返回的 `obs` 是一个 dict，**显式区分 worker 部分与候选项目部分**，方便
DQN/Dueling-DQN 直接用 "worker × candidate" 的方式打分。

| 键 | 形状 | 说明 |
| --- | --- | --- |
| `worker_state` | `(D_w,)` = `(12,)` | 当前 worker 的 12 维静态特征 |
| `candidate_state` | `(K, D_p+D_m)` = `(20, 14)` | 每个候选项目的 10 维 project 特征 + 4 维 match 特征 |
| `valid_mask` | `(K,)` bool | 候选位是否有效（不足 K 个时填 0 并屏蔽） |
| `info` | dict | event_id, worker, timestamp, candidate_projects 列表等元信息 |

### 3.1 Worker 子状态（12 维）

来源：`worker_features.parquet`。
覆盖**质量画像 + 历史画像 + 偏好画像**：

```
worker_quality, worker_total_entries, worker_total_projects,
worker_win_count, worker_finalist_count, worker_avg_score,
worker_win_rate, worker_active_days,
worker_pref_category, worker_pref_sub_category, worker_pref_industry,
worker_category_entropy
```

> 设计要点：
> - 不在 state 中放时间戳本身（避免训练/测试集分布漂移），改用 `worker_active_days` 这种相对量。
> - `worker_pref_*` 直接用类目编码而不是 one-hot：DQN 用 MLP 编码，编码维度可控。

### 3.2 候选 project 子状态（每个 14 维）

每行 = 一个候选项目；前 10 维是 project 特征，后 4 维是 worker-project 匹配特征
（动态计算）。

| 子块 | 列 |
| --- | --- |
| project（10 维） | `project_category, project_sub_category, project_industry_code, project_entry_count, project_total_awards, project_duration_days, project_is_featured, project_average_score, project_worker_count, project_has_winner` |
| match（4 维） | `match_category, match_sub_category, match_industry, match_quality_gap` |

`match_*` 由环境**在线**计算（避免 worker 偏好动态变化时和 parquet 落后）：

```
match_category      = 1{worker_pref_category == project_category, != -1}
match_sub_category  = 1{worker_pref_sub_category == project_sub_category, != -1}
match_industry      = 1{worker_pref_industry == project_industry_code, != -1}
match_quality_gap   = |clip(worker_quality, 0, 1) - clip(project_average_score/5, 0, 1)|
```

### 3.3 扁平化大小

`env.state_dim = D_w + K * (D_p + D_m) = 12 + 20*14 = 292`。

C/D 角色实现 DQN 时可二选一：
- 直接 flatten 成 292 维输入 MLP → 输出 K 个 Q 值；
- 共享 worker encoder + 候选项目 encoder，按 `(s, a_i)` 算 Q，再 mask 无效位。

后者更贴合 "动作集合大小可变" 的众包语义，推荐用 `valid_mask` 屏蔽 padding。

---

## 4. Action（动作）设计

- **动作空间**：离散，`a ∈ {0, 1, ..., K-1}`，对应 `obs["candidate_state"]` 中的某一行。
- **K = 20**（与候选 Top-K 对齐；`event_group` 模式下实际有效位由 `valid_mask` 给出，通常为 3）。
- **无效动作处理**：选到 padding 位（`valid_mask[a] == False`）→ 即时奖励 = 0，但仍然推进 cursor。
  这避免在训练初期 epsilon-greedy 完全无意义；同时让"乱选"自然带来惩罚。

> 简化说明：题目要求"系统只向参与者推荐一个任务"，所以动作就是"在候选集中挑一个"。

---

## 5. Reward（奖励）设计

`src/reward.py` 提供两套 reward，且和 `src/event_stream.py` 的预计算系数完全一致——
即 parquet 中 `reward_worker` / `reward_requester` 列就是 `reward.py` 的输出。

### 5.1 参与者利益 reward (`reward_mode="worker"`)

```
r_worker = label * ( 1.0
                   + 0.5 * worker_quality
                   + 2.0 * winner
                   + 1.0 * finalist )
```

含义：

| 项 | 含义 | 设计动机 |
| --- | --- | --- |
| `label` | 0/1 表示 worker 是否真的接了这个项目 | 反事实约束：只奖励真实参与 |
| `1.0` | 参与基础分 | 任何被采纳的推荐都给一个保底回报 |
| `0.5 * worker_quality` | 给高质量 worker 更高奖励 | quality 越高的人接到任务越能赚到（题目要求"找到更相关、报酬更高的任务"） |
| `2.0 * winner` | 中标加 2 | 中标 ≈ 实际拿到大头报酬 |
| `1.0 * finalist` | 入围加 1 | 入围有部分报酬 |

数值范围：训练集正样本上 mean ≈ 1.44，max = 4.5；负样本恒为 0。

### 5.2 请求者利益 reward (`reward_mode="requester"`)

```
r_requester = label * ( 2.0 * worker_quality
                       + 1.0 * winner )
```

含义：

| 项 | 含义 | 设计动机 |
| --- | --- | --- |
| `2.0 * worker_quality` | 参与者质量直接转嫁给请求者 | 高质量 worker = 更可信的回答 |
| `1.0 * winner` | 选出中标者 | 项目至少选出了优胜者，请求者目标完成 |

数值范围：训练集正样本上 mean ≈ 1.68，max = 3.0。

### 5.3 复用与一致性保证

- `reward.py` 提供 `worker_reward / requester_reward / compute_reward_array`，和
  `WorkerRewardCoef / RequesterRewardCoef` 数据类。
- 系数全部以 `@dataclass(frozen=True)` 暴露，C/D 想做奖励 ablation（比如把
  winner 系数从 2 → 5）只需要：
  ```python
  from src.reward import WorkerRewardCoef
  coef = WorkerRewardCoef(winner=5.0)
  ```
- `test_env.py` 验证：在前 50k 训练样本上，`reward.py` 重算结果和 parquet 中
  `reward_worker / reward_requester` 列的最大绝对误差为 **0.0**。

---

## 6. Next State（下一状态）

- **下一状态 = 时序上的下一个 `(worker, timestamp)` 事件组**——环境用 `_cursor`
  按时间顺序推进，不依赖 agent 的 action（典型 off-policy 离线 RL 设定）。
- 终止状态：所有事件回放完后，返回一个全零观测，`done=True`，`info={"terminal": True}`。

> 注意：这里的"下一状态"和 worker 个人轨迹**无关**——下一条事件可能是另一个
> worker。这是众包平台层视角的 RL：state 描述"当前到达的 worker + 当前候选集"，
> 和参与者级 RL 不同。

---

## 7. Done 与 Episode

- `done=True` 当且仅当 `cursor == len(env)`（事件流耗尽）。
- 一个 episode = 一个完整 split（train ≈ 33.5w 步，val ≈ 6.9w 步，test ≈ 6.9w 步）。
- 训练时通常采取 mini-batch 随机采样 transitions（C 角色实现的 Replay Buffer 负责）。

---

## 8. 两种候选模式

### 8.1 `candidate_mode="event_group"`（默认）

每一步候选集 = A 阶段事件流里 `(worker, timestamp)` 同组的所有行（典型为
1 正 + 2 负，共 3 个 action）。每个候选都有**已知的 reward**——正样本 = parquet
预计算值，负样本 = 0。

**优点**：

- DQN 的 Q-target 信号干净；
- 训练 / 验证 / 测试三个 split 严格不依赖额外数据；
- Random 策略命中率 ≈ 1/3，oracle 收益 ≈ 2× random，对照基线明确。

**用法**：训练 + 主评估都使用此模式。

### 8.2 `candidate_mode="top_k"`

每一步候选集 = `candidates.parquet` 里的 Top-20 活跃项目。除真实选择项目外，
其它候选 reward = 0（反事实未知）。

**用途**：更接近真实推荐场景的 ranking 评估（`Hit@1 / NDCG@K`），E 角色做对比
基线时可调用。已知数据上只有 14.6% 的正样本真实落在 Top-20 内，所以**训练时
不推荐用 top_k 模式**——会把绝大多数 step 的最优 reward 压成 0，导致信号稀疏。

---

## 9. 与 A / C / D / E 的对接

| 角色 | 对接点 | 约定 |
| --- | --- | --- |
| A | parquet 路径 + 列名 | env 直接读 `processed/*.parquet`，列名见第 2 节 |
| C | DQN 训练 | 用 `make_env("train", reward_mode=...)`；从 `obs["candidate_state"]` + `obs["worker_state"]` 构造 Q 输入；用 `obs["valid_mask"]` 做合法动作屏蔽 |
| D | Double / Dueling DQN | 接口同 C，可在 `EnvConfig` 上切换 reward_mode 复用同一 env |
| E | Baseline / 评估 | `env.random_policy(obs)` 直接给 random baseline；`step` 返回的 `info["hit"]` 直接给 Hit@1；切 `candidate_mode="top_k"` 给 Top-K 排序基线 |

---

## 10. Smoke Test 结果

```text
$ python test_env.py
[4] reward.py vs precomputed parquet match
   mode=worker: max abs diff = 0.0000
   mode=requester: max abs diff = 0.0000
[1] mini event_group worker reward
   steps=3927, random_total=2585.08, oracle_total=5597.68, random_hit_rate=0.464
[2] mini event_group requester reward
   total requester reward (random): 2743.89
[3] mini top_k worker reward
   top_k steps=3293, gt-resolvable=1170, random_hit_rate=0.226, total_reward=375.78
[5] make_env('val') – partial rollout
   build steps: 69374 in 2.5s
   first 500 steps: total_reward=228.98, hits=155
All env smoke tests passed.
```

补充全量回放：

```text
test split: 68512 steps in 27s, random_total=33938.1, oracle_total=101807.2,
            random_hit_rate=0.330
train split: 335512 steps built in 12.1s, state_dim=292
```

性能可以接受，C/D 训练时不会被 env 卡瓶颈。

---

## 11. 文件清单

```
CrowdRec-RL/
├── src/
│   ├── env.py              # B-1: CrowdRecEnv + EnvConfig + make_env
│   └── reward.py           # B-2: 两套 reward + 系数 dataclass
├── test_env.py             # B-3: random-policy 端到端 smoke test
└── B_design.md             # 本文档
```
