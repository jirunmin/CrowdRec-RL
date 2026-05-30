
# C 部分交付：基础DQN实现

> 对应 `task_division.md` 中 C 角色的全部交付物：在文件c_basic_dqn中
>
> - `train.py`、`agent_dqn.py`、`test.py` 
> - `basicdqn_plot_curve.py`
> - best模型保存`basic_dqn_best_requester_model.pth`、`basic_dqn_best_worker_model.pth`
> - 训练数据 `train_log_worker_per_5000_steps.csv`、`train_log_worker_per_episode.csv`、     `train_log_requester_per_5000_steps.csv`、`basic_dqn_best_requester_model.pth`
> - 训练曲线 `worker_episode_loss.png`（以episode为单位的折线图）、`worker_episode_reward.png`（以episode为单位的reward折线图）、`worker_step_loss_epsilon.png`（以全局step为单位的epsilon、loss折线图）、`worker_step_reward.png`（以全局step为单位的reward折线图）；`requester_episode_loss.png`（以episode为单位的折线图）、`requester_episode_reward.png`（以episode为单位的reward折线图）、`requester_step_loss_epsilon.png`（以全局step为单位的epsilon、loss折线图）、`requester_step_reward.png`（以全局step为单位的reward折线图）
> - 测试结果 Worker score: 101804.3550、Requester score: 123326.7930。柱状图：`final_comparison.png`

---

## 1. 整体流程概览

项目采用标准 DQN 流程：
1. 环境给出观测 `obs`（包含 `worker_state`、`candidate_state`、`valid_mask`）。
2. `obs_to_state` 将观测拼接为一维状态向量。
3. `DQNAgent.select_action` 基于 ε-greedy 与 `valid_mask` 选动作。
4. 与环境交互后，把转移样本写入经验池。
5. 经验池达到预热阈值后，开始 `update()` 训练 Q 网络。
6. 周期性硬更新目标网络，稳定训练。
7. 每个 episode 结束在验证集评估，若更优则保存最优模型。

---

## 2. 三个类的设计（`agent_dqn.py`）

## 2.1 `ReplayBuffer`

### 职责
- 存储离线经验样本 `(state, action, reward, next_state, done)`
- 随机采样 mini-batch，打破样本时间相关性

### 关键实现
- 底层使用 `deque(maxlen=capacity)` 控制容量上限
- `push(...)`：写入新样本，自动淘汰最旧样本
- `sample(batch_size)`：随机无放回采样，返回张量
- `__len__()`：返回当前样本数量

### 输入/输出约定
- `state/next_state` 转 `np.float32`
- `actions` 输出为 `LongTensor`，并 `unsqueeze(1)` 以适配 `gather`
- `rewards/dones` 输出为 `FloatTensor` 且形状 `[B, 1]`

---

## 2.2 `QNetwork`

### 职责
- 从状态向量映射到每个动作的 Q 值

### 结构
- 默认多层感知机（MLP）：
  - 输入层：`state_dim`
  - 隐藏层：`[128, 128]`，激活函数 `ReLU`
  - 输出层：`action_dim`



### 前向
- `forward(x)` 返回 `self.net(x)`
- 支持 batch 输入（形状通常 `[B, state_dim]`）

---

## 2.3 `DQNAgent`

### 职责
封装 DQN 训练与推理核心逻辑：
- 动作选择（含动作 mask）
- 经验存储
- 网络更新
- 目标网络同步
- 模型保存/加载

### 核心成员
- `q_net`：主网络（用于行为与学习）
- `target_net`：目标网络（用于 TD 目标）
- `optimizer`：`Adam`
- `replay_buffer`：经验池
- `epsilon`：探索率
- `device`：`cpu` 或 `cuda`

### 关键方法

#### 1) `select_action(state, valid_mask=None, eval_mode=False)`
- 训练时：ε-greedy，带衰减率
- 测试时：贪心（`eval_mode=True`）
- `valid_mask` 用于屏蔽非法动作：
  - 将非法动作 Q 值设为 `-inf`
  - 在合法动作中 `argmax`

#### 2) `update()`
- 前提：经验池长度不少于 `batch_size`
- TD 目标：
\[
y = r + \gamma \max_{a'} Q_{\theta^-}(s', a') \cdot (1-d)
\]
- 当前值：
\[
Q_{\theta}(s,a)
\]
- 损失：`MSELoss(current_q, target_q)`
- 反向传播 + 梯度裁剪（`clip_grad_norm_`）
- `epsilon` 按衰减系数递减到下限
- 每 `target_update_freq` 步进行一次硬更新：
  - `target_net <- q_net`

#### 3) `store_transition(...)`
- 写入经验池

#### 4) `save_model(path)` / `load_model(path)`
- 只保存/加载在线网络权重
- 加载后同步目标网络

---

## 3. 训练脚本设计（`train.py`）

## 3.1 关键固定配置
- `WARMUP_STEPS = 3000`：经验池预热阈值
- `SEED = 42`：随机种子
- `CANDIDATE_MODE = "event_group"`

## 3.2 训练入口参数
- `--device`: `auto | cpu | cuda`（默认 `auto`）
- `--episodes`: 默认 `3`

## 3.3 设备与 batch size 策略
- `auto` 时自动选择可用 GPU，否则 CPU
- CPU 训练时自适应增大 batch size：
  - 核心数 >= 16：至少 256
  - 核心数 >= 8：至少 128
  - 其它：至少 64

## 3.4 Agent 初始化参数
当前训练脚本给定：
- `lr=5e-4`
- `epsilon_decay=0.995`
- `batch_size=resolved_batch_size`（由设备策略确定）

Agent 其余默认超参（来自 `DQNAgent`）：
- `gamma=0.99`
- `epsilon_start=1.0`采用线性epsilon
- `buffer_capacity=400000`
- `target_update_freq=500`
- `grad_clip_norm=20.0`

## 3.5 训练过程细节
1. 每个 `reward_mode`（`worker`、`requester`）分别训练
2. episode 内循环：交互、存经验、预热后更新网络
3. 预热阶段打印经验池进度（每 5%）
4. 每累计 5000 环境步打印一次全局步数
5. 每个 episode 后：
   - 在 `val` 集评估，一次跑3个episode取平均值
   - 若验证 reward 创新高，保存最优模型
6. 保存训练日志到：
   - `train_log_worker_per_5000_steps.csv`、`train_log_worker_per_episode.csv`
   - `train_log_requester_per_5000_steps.csv`、`train_log_requester_per_episode.csv`

## 3.6 输出模型
- `basic_dqn_best_worker_model.pth`
- `basic_dqn_best_requester_model.pth`

---

## 4. 测试脚本设计（`test.py`）

## 4.1 入口参数
- `--device`: `auto | cpu | cuda`（默认 `auto`）

## 4.2 测试流程
1. 创建 `split="test"` 环境
2. 实例化 `DQNAgent` 并加载模型
3. `eval_mode=True` 下执行贪心策略
4. 累加整局 reward，输出最终分数
5. 分别测试 worker / requester 模型
6. 生成柱状图：`final_comparison.png`

## 4.3 测试输出
- 控制台打印：
  - 实际评估设备
  - `Worker score`
  - `Requester score`
- 图像文件：`final_comparison.png`

---

## 5. Q 网络的基础设计说明（简明）

当前 Q 网络是一个轻量级 MLP，适合中小规模离散动作场景：
- 两层 128 隐藏单元可以在表达能力与训练稳定性间取得平衡。
- 输出层直接预测每个动作 Q 值，便于配合 `argmax` 选动作。
- 在本项目中，`valid_mask` 机制保证只在合法候选动作中决策。

若后续需要提升效果，可考虑：
- 增大隐藏维度（如 256/256）
- Double DQN（缓解 Q 值过估计）
- Dueling 结构（价值/优势分解）
- Prioritized Replay（提升样本效率）

---

## 6. 当前参数小结（可复现实验）

推荐命令：

```bash
python train.py --device cpu --episodes 6
python test.py --device cpu
```

该配置下：
- 训练 6 个 episode（每个 reward_mode 各 6 轮）
- 经验池预热 3000 样本后开始更新
- 每 5000 步打印一次全局步数
- 每个 episode 结束依据验证集表现保存最优模型
