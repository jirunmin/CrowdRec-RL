# CrowdRec-RL：众包任务推荐的强化学习

## 0. 快速开始

```bash
# 创建 conda 环境并安装依赖
conda create -n crowdrec python=3.10 -y
conda activate crowdrec
pip install -r requirements.txt

# 重跑数据预处理（如需，约 2 分钟）
python main_preprocess.py --output_dir processed --neg_ratio 2.0 --top_k 20

# 查看数据统计
python plot_stats.py            # 生成 figures/*.png + 终端打印摘要
```

---

## 1. 项目目标

用 DQN 做众包任务推荐，同时优化两方利益：
- **参与者视角**：让 worker 找到更相关、报酬更高的任务
- **请求者视角**：让 project 获得更多、更高质量的回答

---

## 2. 原始数据（不要动）

| 文件/目录 | 内容 | 规模 |
|-----------|------|------|
| `worker_quality.csv` | worker_id → quality (0-100, -1=缺失) | 1,807 人 |
| `project_list.csv` | project_id → entry_count | 2,501 个项目 |
| `project/project_{id}.txt` | 项目详情 JSON：category, industry, start_date, deadline, awards 等 | 5,307 个 |
| `entry/entry_{pid}_{offset}.txt` | 回答记录（每 24 条分一页）：`author`(worker_id), `entry_created_at`, `winner`, `score` | 22,805 个文件, 5,295 个 unique 项目 |

### 关键字段

**entry JSON 中**：
- `author` → 就是 worker_id（不是 `worker`）
- `entry_created_at` → 提交时间，我们**近似作为 worker 到达时间**
- `winner` / `finalist` / `score` → 质量信号

---

## 3. 预处理流程（已完成，结果在 `processed/`）

```
raw data ──→ [data_loader] ──→ [preprocessing] ──→ [features] ──→ [event_stream] ──→ [dataset split] ──→ processed/*.parquet
```

### 3.1 加载 (`src/data_loader.py`)

- 扫描 `entry/` 文件名发现 5,295 个有交互的项目
- 合并 `project_list.csv` 中的项目作为候选池扩充
- 读取所有 project 详情 + entry 记录
- **注意**：entry JSON 的 worker 字段叫 `author`，已映射为 `worker`

### 3.2 清洗 (`src/preprocessing.py`)

- **worker quality 缺失值**：采用两步策略——先用中位数 78 做基础填充；后续 Step 3.5 通过 ML 预测覆盖（见下文）
- 过滤 status 无效的项目（draft/cancelled 等）
- 移除 entry_created_at 晚于项目 deadline 的异常记录
- **扩展 worker_df**：entry 中实际出现 11,865 个 worker，原始 csv 仅 1,807 个，自动补全新 worker 行

### 3.3 特征工程 (`src/features.py`)

**Worker 特征 (16维)**：quality, total_entries, total_projects, win_count, win_rate, finalist_count, avg_score, pref_category, pref_sub_category, pref_industry, category_entropy, active_days, first/last_entry

**Project 特征 (13维)**：category, sub_category, industry_code, entry_count, total_awards, duration_days, is_featured, average_score, worker_count, has_winner

**匹配特征 (4维，向量化计算)**：match_category, match_sub_category, match_industry, match_quality_gap（|worker_quality - project_avg_score/5|）

### 3.3.5 Worker Quality 预测 (`src/quality_predictor.py`)  

entry 中有 11,865 个 worker，但仅 1,653 个有 quality 标签。训练 GBDT 回归器，用**行为特征预测 quality**：

| 指标 | 值 |
|------|-----|
| 模型 | GradientBoostingRegressor |
| 输入特征 | total_entries, win_rate, finalist_count, avg_score, active_days 等 11 维 |
| 有标签训练集 | 1,653 workers |
| 预测目标 | 10,212 workers |
| CV MAE (5-fold) | **3.07** (quality 范围 0-100) |
| CV R² | **0.634** |

**Top 3 重要特征**：finalist_count (0.30) > active_days (0.21) > win_count (0.19)

`quality` 默认使用 `--quality_mode predict` 模式下的 ML 预测值（列 `worker_quality`），切换 `--quality_mode median` 则用中位数填充值。

### 3.4 RL 事件流 (`src/event_stream.py`)

- **正样本** (label=1)：每条 entry 是一个事件——worker 在 timestamp 真实参与了 project
- **负样本** (label=0)：对每个正样本，从当时活跃项目中随机采样 2 个作为负样本（worker 看到但没选）
- 事件按 timestamp 排序，总计 1,393,137 条（正 474,961 + 负 918,176）

**Reward 定义**：
```
reward_worker    = 1.0(参与) + 0.5×quality + 2.0×winner + 1.0×finalist
reward_requester = 2.0×quality + 1.0×winner
```


### 3.5 数据集划分 (`src/dataset.py`)

**严格按时序**切分，不随机 shuffle：

| 数据集 | 条数 | 时间范围 |
|--------|------|----------|
| Train | 975,195 (70%) | 2008-05 ~ 2018-03 |
| Val | 208,971 (15%) | 2018-03 ~ 2018-09 |
| Test | 208,971 (15%) | 2018-09 ~ 2019-03 |

**候选集生成**：为每个正样本事件计算当时活跃的 Top-20 候选项目，按 `urgency + featured_bonus + award_score` 排序。保存在 `candidates.parquet` 中。

---

## 4. 处理后的数据格式

### 4.1 `train_events.parquet` / `val_events.parquet` / `test_events.parquet`

每行 = 一个 (worker, project) 对 + 全部特征 + label + reward。共 **43 列**，详细含义见下方各表。

**特征列（完整列表）**：

#### Worker 特征（16维）—— 来自 `worker_features.parquet` 与事件表 merge

| 列名 | 类型 | 含义 |
|------|------|------|
| `worker_quality` | float [0,1] | Worker 质量评分（默认 `--quality_mode predict` 为 ML 预测值；`--quality_mode median` 为中位数填充值） |
| `worker_quality_raw` | float [0,100] | Worker 原始质量评分，-1 = 无标签（用 `worker_quality_pred` 覆盖） |
| `worker_total_entries` | float | 该 worker 历史上总共提交的回答数 |
| `worker_total_projects` | float | 该 worker 历史上参与过的不同项目数 |
| `worker_win_count` | float | 累计中标次数（winner=True） |
| `worker_win_rate` | float [0,1] | 中标率 = win_count / total_entries |
| `worker_finalist_count` | float | 累计入围次数（finalist=True） |
| `worker_avg_score` | float | 历史平均得分 |
| `worker_pref_category` | float | 最常参与的项目大类（category 的 mode），-1=无历史 |
| `worker_pref_sub_category` | float | 最常参与的项目子类（sub_category 的 mode），-1=无历史 |
| `worker_pref_industry` | float | 最常参与的行业编码（industry_code 的 mode），-1=无历史 |
| `worker_category_entropy` | float | Worker 参与类别的熵（越高=兴趣越分散，越低=越专注） |
| `worker_active_days` | float | 从首次参与到最后参与的天数跨度 |
| `worker_first_entry` | datetime | 首次参与时间 |
| `worker_last_entry` | datetime | 最后参与时间 |

> **注意**：`worker_total_entries` / `worker_win_count` 等动态特征是**从全部历史数据统计的全局值**，而非事件发生时刻的快照。如需时序增量特征需在环境中自行维护。

#### Project 特征（13维）—— 来自 `project_features.parquet` 与事件表 merge

| 列名 | 类型 | 含义 |
|------|------|------|
| `project_category` | int | 项目大类编码（如 7=Logo设计, 23=包装设计） |
| `project_sub_category` | int | 项目子类编码 |
| `project_industry_code` | int | 行业编码（36个行业，如 healthcare=某码, tech=某码） |
| `project_entry_count` | int | 该项目收到/期望的回答总数 |
| `project_total_awards` | float | 项目总奖金（美元） |
| `project_duration_days` | int | 项目持续天数 = deadline - start_date |
| `project_is_featured` | int (0/1) | 是否为精选/推广项目 |
| `project_average_score` | float | 项目所有回答的平均得分（约 0-5 分制） |
| `project_worker_count` | int | 参与该项目的不同 worker 数量 |
| `project_has_winner` | int (0/1) | 是否已选出中标者 |
| `project_start_date` | datetime | 项目开始时间 |
| `project_deadline` | datetime | 项目截止时间 |

#### 匹配特征（4维）—— 向量化计算，描述 worker 与 project 的匹配程度

| 列名 | 类型 | 含义 |
|------|------|------|
| `match_category` | float (0/1) | worker 偏好大类 == 项目大类？ |
| `match_sub_category` | float (0/1) | worker 偏好子类 == 项目子类？ |
| `match_industry` | float (0/1) | worker 偏好行业 == 项目行业？ |
| `match_quality_gap` | float [0,1] | \|worker_quality - project_avg_score/5\|，越小越匹配 |

#### 标签与奖励列

| 列名 | 类型 | 含义 |
|------|------|------|
| `label` | int (0/1) | 1=正样本（worker 真实参与了该项目），0=负样本（未参与） |
| `winner` | bool | 该回答是否中标（仅正样本有效） |
| `finalist` | bool | 该回答是否入围（仅正样本有效） |
| `score` | int | 回答评分（仅正样本有效） |
| `reward_worker` | float | Worker 视角即时奖励 = 1.0 + 0.5×quality + 2.0×winner + 1.0×finalist |
| `reward_requester` | float | Requester 视角即时奖励 = 2.0×quality + 1.0×winner |

#### 辅助列

| 列名 | 类型 | 含义 |
|------|------|------|
| `event_id` | int | 全局事件序号，按时序递增，跨 train/val/test 唯一 |
| `worker` | int | Worker ID（可 join `worker_features.parquet`） |
| `project_id` | int | Project ID（可 join `project_features.parquet`） |
| `timestamp` | datetime | 事件时间（worker 到达时间，UTC 时区） |
| `day_index` | int | 从首个事件起的天数偏移量 |
| `split` | str | "Train" / "Val" / "Test" |

### 4.2 `worker_features.parquet` / `project_features.parquet`

Worker/Project 的**静态特征查找表**，用于环境中构造 state。

### 4.3 `candidates.parquet`

仅包含正样本事件，多了 `candidate_projects` 列：
- 类型：`list[int]`，长度 0~20
- 含义：该事件时刻，当时活跃的 Top-20 候选项目 ID 列表

### 4.4 `stats.json`

```json
{"n_train":975195, "n_val":208971, "n_test":208971, "n_workers":11865, "n_projects":5131}
```

---

## 5. 实现 RL 环境 (`env.py`)

### 5.1 环境基本框架

```python
class CrowdRecEnv:
    def __init__(self, events_df, worker_features, project_features,
                 candidates_df=None, reward_mode="worker"):
        """
        reward_mode: "worker" 或 "requester"
        """
        self.events = events_df.sort_values("timestamp")
        self.current_idx = 0
        ...

    def reset(self):
        """重置到数据集起点，返回第一个 state"""
        ...

    def step(self, action):
        """
        action: 推荐的 project_id (int)
        返回: (next_state, reward, done, info)
        """
        ...
```

### 5.2 State 设计建议

State 是当前决策点 agent 能看到的信息。建议结构：

```
state = {
    # 当前 worker 特征（静态 + 动态）
    "worker_quality": 0.82,
    "worker_total_entries": 45,
    "worker_pref_category": 7,
    "worker_win_rate": 0.12,
    ...

    # 候选池中每个 project 的特征（向量）
    "candidate_features": [[...], [...], ...],   # shape: (K, N_project_features)

    # 全局上下文
    "day_index": 1234,
    "n_active_projects": 20,
}
```

**状态转移**：当前事件 → 下一个事件。worker 特征会随历史累积而更新（如 `worker_total_entries` 增加）。

### 5.3 Action 设计

- **离散动作空间**：大小为 K（候选项目数），每个动作对应选择一个 project_id
- 如果候选池大小 < K，用 mask 遮蔽无效动作
- 只从 `candidate_projects` 列表中选（保证逻辑合理）

### 5.4 Reward 设计

两个模式，分别对应题目要求：

**模式 1: worker_reward（最大化参与者利益）**
```python
# 直接用数据中的 reward_worker 列
reward = event["reward_worker"]  # 范围 [0, 4.5]
```

**模式 2: requester_reward（最大化请求者利益）**
```python
# 直接用数据中的 reward_requester 列
reward = event["reward_requester"]  # 范围 [0, 3.0]
```

也可以自行设计更复杂的 reward 函数，但必须保证：
- 正样本 reward > 负样本 reward（正样本 reward 通常 > 1.0，负样本 = 0）

### 5.5 Next State & Done

- **Next State**：下一个事件的 state
- **Done**：到达数据集末尾（最后一个事件被处理后）

### 5.6 数据读取方式

```python
import pandas as pd

# 加载
train = pd.read_parquet("processed/train_events.parquet")
workers = pd.read_parquet("processed/worker_features.parquet")
projects = pd.read_parquet("processed/project_features.parquet")
candidates = pd.read_parquet("processed/candidates.parquet")

# 按时间排序
train = train.sort_values("timestamp").reset_index(drop=True)

# 逐行遍历
for idx, event in train.iterrows():
    state = build_state(event, workers, projects, candidates)
    action = agent.act(state)
    # 根据 action 对应的 project_id 判断是否命中正样本
    ...
```

### 5.7 Random 策略验证要求

必须能在环境中跑通 Random 策略：
```python
total_reward = 0
state = env.reset()
done = False
while not done:
    action = np.random.choice(state["candidate_ids"])  # 随机选一个候选
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    state = next_state
print(f"Random policy total reward: {total_reward}")
```

---

## 6. 项目文件结构

```
CrowdRec-RL/
├── processed/                 # ← DQN输入数据（已生成）
│   ├── train_events.parquet   # 训练集（975k 条）
│   ├── val_events.parquet     # 验证集（209k 条）
│   ├── test_events.parquet    # 测试集（209k 条）
│   ├── worker_features.parquet
│   ├── project_features.parquet
│   ├── candidates.parquet     # Top-20 候选
│   └── stats.json
├── figures/                   # 数据统计图表（8 张）
├── src/                       # 预处理代码（已完成，一般不需改动）
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── quality_predictor.py   # Worker quality ML 预测
│   ├── event_stream.py
│   └── dataset.py
├── main_preprocess.py         # 预处理入口
├── plot_stats.py              # 统计图表入口
├── sample_read_data.py        # 助教参考代码（已修正 header bug）
├── requirements.txt           # Python 依赖（pip install -r requirements.txt）
├── worker_quality.csv
├── project_list.csv
├── project/                   # 原始项目文件
└── entry/                     # 原始 entry 文件
```

---

## 7. 数据特点

1. **时序严格**：数据不能 shuffle，train/val/test 是按时间切好的，训练时也要按时间顺序
2. **前几个事件候选为空**：最早的事件发生时，平台上还没有活跃项目，candidates=[]，需要在代码中处理
3. **负样本 reward=0**：负样本的 reward_worker 和 reward_requester 都是 0，正样本 > 1.0
4. **中标率极低**：仅 1.0% 的 entry 中标（winner=True），reward 分布有长尾
5. **worker 特征会动态变化**：worker_total_entries 等特征随历史累积而增长，环境中需要考虑是否更新这些特征
6. **category 匹配有区分力**：正样本 match_category=0.192，负样本=0.119，说明类别匹配是有效特征
