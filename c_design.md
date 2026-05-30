# C 部分交付：基础 DQN 实现（当前代码同步版）

覆盖以下核心文件，在c_basic_dqn中：
- `train.py`
- `agent_dqn.py`
- `test.py`
- `basicdqn_plot_curve.py`
- best模型保存`basic_dqn_best_requester_model.pth`、`basic_dqn_best_worker_model.pth`
- 训练数据 `train_log_worker_per_5000_steps.csv`、`train_log_worker_per_episode.csv`、     `train_log_requester_per_5000_steps.csv`、`basic_dqn_best_requester_model.pth`
- 训练曲线 `worker_episode_loss.png`（以episode为单位的折线图）、`worker_episode_reward.png`（以episode为单位的reward折线图）、`worker_step_loss_epsilon.png`（以全局step为单位的epsilon、loss折线图）、`worker_step_reward.png`（以全局step为单位的reward折线图）；`requester_episode_loss.png`（以episode为单位的折线图）、`requester_episode_reward.png`（以episode为单位的reward折线图）、`rworker_episode_success_rate.png`(以episode为单位的success_rate折线图)、`worker_step_success_rate.png`（以全局step为单位的success_rate折线图）；`requester_step_loss_epsilon.png`（以全局step为单位的epsilon、loss折线图）、`requester_step_reward.png`（以全局step为单位的reward折线图）、`requester_episode_success_rate.png`(以episode为单位的success_rate折线图)、`requester_step_success_rate.png`（以全局step为单位的success_rate折线图）
测试结果 Worker score: 65615.6864、Requester score: 80174.7314。柱状图：`final_comparison.png`

并说明与环境侧 `fastenv` 的交互方式（训练/验证/测试统一使用归一化环境）。

---

## 1. 总体流程

当前项目采用标准 DQN 训练闭环：
1. 环境返回观测 `obs`，包含：
   - `worker_state`
   - `candidate_state`
   - `valid_mask`
2. `obs_to_state` 将 `worker_state` 与 `candidate_state` 展平并拼接为一维向量。
3. Agent 使用带 `valid_mask` 的 ε-greedy 选择动作。
4. 与环境交互后，将转移存入经验回放池。
5. 经验池达到预热阈值后开始参数更新。
6. 周期性同步目标网络。
7. 每个 episode 结束在验证集评估：
   - `val_reward`
   - `val_success_rate`
8. 基于验证指标执行早停，并在 `val_reward` 提升时保存最佳模型。

---

## 2. 环境接入（train/test 与 fastenv）

`train.py` 与 `test.py` 均已切换到：
- `from src.fastenv import make_fast_env`

并统一使用：
- `normalize_features=True`

这意味着：
- 训练/验证/测试使用同一套状态预处理口径；
- 归一化统计由环境侧处理（非 `train.py` 内部处理）；
- `train.py` 本身仅做状态拼接，不自行计算均值/标准差。

---

## 3. `agent_dqn.py` 设计

### 3.1 ReplayBuffer

职责：存储并随机采样经验。

- 数据结构：`deque(maxlen=capacity)`
- 单条样本：
  - `(state, action, reward, next_state, next_valid_mask, done)`
- 采样输出：
  - `states`: `FloatTensor [B, state_dim]`
  - `actions`: `LongTensor [B, 1]`
  - `rewards`: `FloatTensor [B, 1]`
  - `next_states`: `FloatTensor [B, state_dim]`
  - `next_valid_masks`: `BoolTensor [B, action_dim]`
  - `dones`: `FloatTensor [B, 1]`

### 3.2 QNetwork

- 结构：MLP，默认隐藏层 `[128, 128]` + ReLU
- 输入：`state_dim`
- 输出：`action_dim`（每个动作一个 Q 值）

### 3.3 DQNAgent

核心能力：
- 自动设备选择（`auto` -> CUDA 可用则 GPU）
- 线性 ε 衰减（按全局步数）
- 合法动作掩码决策（`valid_mask`）
- DQN 更新（在线网 + 目标网）
- 梯度裁剪
- 模型保存/加载

#### 动作选择
- 训练：ε-greedy
- 评估：贪心（`eval_mode=True`）
- 若提供 `valid_mask`：仅在合法动作中取 `argmax`

#### 更新公式
\[
y = r + \gamma \cdot \max_{a'} Q_{target}(s',a') \cdot (1-d)
\]

- 对 `next_valid_masks` 做非法动作屏蔽（置 `-inf`）
- 若某样本下一步无合法动作，则该样本 `next_q` 置 0
- 损失：`MSE(current_q, target_q)`

---

## 4. `train.py` 训练脚本（当前实现）

### 4.1 关键常量

- `WARMUP_STEPS = 3000`
- `STEP_LOG_INTERVAL = 5000`
- `REWARD_SCALE = 20.0`
- `EPSILON_DECAY_STEPS = 1_675_000`
- `TARGET_UPDATE_FREQ = 2000`
- `EVAL_EPISODES = 3`
- `SEED = 42`
- `CANDIDATE_MODE = "event_group"`

### 4.2 早停机制（已改为验证指标）

- `EARLY_STOP_PATIENCE = 5`
- `VAL_METRIC_MIN_DELTA = 0.002`
- `EARLY_STOP_METRIC = "val_reward"`（可切为 `val_success_rate`）

逻辑：
- 每个 episode 结束后评估验证集；
- 取监控指标：
  - 若 `EARLY_STOP_METRIC = "val_reward"`，监控 `val_reward`；
  - 若 `EARLY_STOP_METRIC = "val_success_rate"`，监控 `val_success_rate`；
- 若本轮提升不超过 `VAL_METRIC_MIN_DELTA`，记一次“无明显提升”；
- 连续达到 `EARLY_STOP_PATIENCE` 后提前停止。

### 4.3 训练流程

1. 创建训练环境：`make_fast_env(split="train", normalize_features=True, ...)`
2. 初始化 Agent（含 batch size 与设备策略）
3. episode 循环：
   - 交互采样
   - 回放池预热（输出 warmup 进度）
   - 预热后调用 `agent.update()`
   - 每 5000 全局步记录 step 日志
4. episode 结束后：
   - 计算训练侧 `reward/loss/success_rate`
   - 调用 `evaluate_model(..., split="val")`
   - 如 `val_reward` 提升则保存最佳模型
   - 执行验证指标早停判定

### 4.4 success_rate 指标

当前已在训练与验证侧启用：
- 统计口径：
  - 当 `ground_truth_index >= 0` 记为有效 step；
  - `info["hit"]` 计入命中；
  - `success_rate = hits / valid_steps`。

### 4.5 日志输出

#### Step 日志（每 5000 步）
文件：`train_log_{reward_mode}_per_5000_steps.csv`

字段包含：
- `global_step`
- `episode`
- `episode_inner_step`
- `buffer_size`
- `epsilon`
- `episode_reward_so_far`
- `episode_avg_loss_so_far`
- `episode_success_rate_so_far`

#### Episode 日志
文件：`train_log_{reward_mode}_per_episode.csv`

字段包含：
- `reward`
- `loss`
- `success_rate`
- `eps`

### 4.6 训练参数与 CLI

支持参数：
- `--device {auto,cpu,cuda}`
- `--episodes <int>`
- `--reward-mode {worker,requester,both}`
- `--early-stop-metric {val_reward,val_success_rate}`

说明：
- `both` 会顺序训练 `worker` 与 `requester`。

---

## 5. `test.py` 评估脚本（当前实现）

### 5.1 环境

- 使用 `make_fast_env(split="test", normalize_features=True, ...)`
- 与训练阶段预处理口径一致。

### 5.2 测试流程

1. 加载指定模型权重；
2. `eval_mode=True` 贪心选动作；
3. 累加整局奖励为最终得分；
4. 分别评估：
   - `basic_dqn_best_worker_model.pth`
   - `basic_dqn_best_requester_model.pth`
5. 输出柱状图：`final_comparison.png`

### 5.3 CLI

- `--device {auto,cpu,cuda}`

---

## 6. 当前实现特性与说明

1. **训练脚本不自行归一化**：归一化由环境层处理。
2. **验证指标早停**：相比按训练 loss 早停，更贴近泛化目标。
3. **模型保存策略**：当前按 `val_reward` 最优保存 best checkpoint。
4. **动作合法性约束**：训练/评估全程基于 `valid_mask`，避免选择非法候选。

---

## 7. 复现示例

训练（两个 reward 模式）：

```bash
python train.py --device cpu --episodes 20 --reward-mode both --early-stop-metric val_reward
```

若希望按成功率早停：

```bash
python train.py --device cpu --episodes 20 --reward-mode both --early-stop-metric val_success_rate
```

测试：

```bash
python test.py --device cpu
```
