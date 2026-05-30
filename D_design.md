# D 部分交付：Double DQN 实现

> 对应 D 角色的 Double DQN 交付物，代码位于 `d_double_dqn/`。

## 1. 实现文件

- `d_double_dqn/agent_double_dqn.py`
  - `ReplayBuffer`
  - `QNetwork`
  - `DoubleDQNAgent`
- `d_double_dqn/train.py`
  - 训练 worker / requester 两种 reward
  - 保存最优模型与训练日志
- `d_double_dqn/test.py`
  - 在 test split 上评估模型
  - 输出 cumulative reward 与 hit rate
- `d_double_dqn/doubledqn_plot_curve.py`
  - 根据训练日志绘制 reward / loss / epsilon 曲线

## 2. 和基础 DQN 的核心区别

基础 DQN 的 target 是：

```text
y = r + gamma * max_a Q_target(s_next, a)
```

它使用同一个 target network 同时做动作选择和动作估值，容易产生 Q 值过估计。

Double DQN 将这两步拆开：

```text
a_next = argmax_a Q_online(s_next, a)
y = r + gamma * Q_target(s_next, a_next)
```

也就是：

1. 在线网络 `q_net` 负责在下一状态选择动作。
2. 目标网络 `target_net` 负责估计该动作的 Q 值。
3. `valid_mask` 会同时作用在下一状态动作选择上，避免 padding 动作参与 argmax。

对应实现位于：

```python
online_next_q_all = self.q_net(next_states)
masked_online_next_q[next_valid_masks] = online_next_q_all[next_valid_masks]
next_actions = masked_online_next_q.argmax(dim=1, keepdim=True)

target_next_q_all = self.target_net(next_states)
next_q = target_next_q_all.gather(1, next_actions)
target_q = rewards + self.gamma * next_q * (1 - dones)
```

## 3. 训练流程

训练流程和 C 部分基础 DQN 保持一致，便于做对比：

1. `make_env(...)` 构造离线推荐环境。
2. `obs_to_state(...)` 拼接 `worker_state` 与 flatten 后的 `candidate_state`。
3. `DoubleDQNAgent.select_action(...)` 使用 epsilon-greedy，并用 `valid_mask` 屏蔽非法动作。
4. transition 存入 replay buffer。
5. replay buffer 预热后调用 `agent.update()`。
6. 每隔 `target_update_freq` 次网络更新，同步一次 target network。
7. 每个 episode 后在 val split 上评估，并保存最优模型。

## 4. 推荐命令

从项目根目录执行：

```bash
python d_double_dqn/train.py --reward-mode both --episodes 6 --device cpu
python d_double_dqn/test.py --reward-mode both --device cpu
python d_double_dqn/doubledqn_plot_curve.py --reward-mode worker
python d_double_dqn/doubledqn_plot_curve.py --reward-mode requester
```

如果沿用 C 部分设置，默认 `gamma=0.99`。由于当前环境更接近离线单步推荐排序问题，也可以做一个消融实验：

```bash
python d_double_dqn/train.py --reward-mode both --episodes 6 --device cpu --gamma 0.0
```

`gamma=0.0` 时 target 退化为即时 reward，更适合动作不会影响下一条事件的离线数据设定；`gamma=0.99` 则便于和 C 部分基础 DQN 的配置直接对比。

## 5. 输出文件

训练后默认输出到 `d_double_dqn/`：

- `double_dqn_best_worker_model.pth`
- `double_dqn_best_requester_model.pth`
- `train_log_double_dqn_worker_per_5000_steps.csv`
- `train_log_double_dqn_worker_per_episode.csv`
- `train_log_double_dqn_requester_per_5000_steps.csv`
- `train_log_double_dqn_requester_per_episode.csv`
- `double_dqn_worker_step_loss_epsilon.png`
- `double_dqn_worker_step_reward.png`
- `double_dqn_worker_episode_reward.png`
- `double_dqn_worker_episode_loss.png`
- `double_dqn_requester_step_loss_epsilon.png`
- `double_dqn_requester_step_reward.png`
- `double_dqn_requester_episode_reward.png`
- `double_dqn_requester_episode_loss.png`
- `double_dqn_final_comparison.png`

## 6. 注意事项

- 需要先安装 `requirements.txt` 中的依赖，尤其是 `torch` 和 `pyarrow`。
- 新脚本内部使用项目根目录定位 `src/` 和 `processed/`，从项目根目录或 `d_double_dqn/` 目录启动都可以。
- 当前网络结构仍沿用 C 部分的 slot-based MLP：输入为整个候选集合 flatten 后的状态，输出为固定 action slot 的 Q 值。若要进一步提升 D 部分，可以继续改成共享的 `Q(worker, candidate_i)` 候选打分结构。
