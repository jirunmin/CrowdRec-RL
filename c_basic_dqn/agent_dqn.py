import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_valid_mask, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        next_valid_mask = np.array(next_valid_mask, dtype=bool)
        self.buffer.append((state, action, reward, next_state, next_valid_mask, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_valid_masks, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        next_valid_masks = torch.as_tensor(np.array(next_valid_masks), dtype=torch.bool)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        return states, actions, rewards, next_states, next_valid_masks, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device=None,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=2000_000,
        buffer_capacity=400000,
        batch_size=64,
        target_update_freq=1000,
        grad_clip_norm=20.0,
    ):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.epsilon = self.epsilon_start

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        self.step_count = 0
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def set_epsilon_by_step(self, global_step: int):
        progress = min(max(global_step, 0) / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, valid_mask=None, eval_mode=False):
        if valid_mask is None:
            valid_indices = np.arange(self.action_dim, dtype=np.int64)
        else:
            mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if mask.size != self.action_dim:
                raise ValueError(f"valid_mask size ({mask.size}) != action_dim ({self.action_dim})")
            valid_indices = np.flatnonzero(mask)
            if valid_indices.size == 0:
                valid_indices = np.arange(self.action_dim, dtype=np.int64)
        if (not eval_mode) and (np.random.rand() < self.epsilon):
            return int(np.random.choice(valid_indices))
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state = state.to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state).squeeze(0)

        if valid_mask is not None:
            valid_mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device).view(-1)
            if valid_mask_tensor.numel() != self.action_dim:
                raise ValueError(
                    f"valid_mask size ({valid_mask_tensor.numel()}) != action_dim ({self.action_dim})"
                )

            if valid_mask_tensor.any():
                masked_q = torch.full_like(q_values, -torch.inf)
                masked_q[valid_mask_tensor] = q_values[valid_mask_tensor]
                return int(torch.argmax(masked_q).item())

        return int(torch.argmax(q_values).item())

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, next_valid_masks, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_valid_masks = next_valid_masks.to(self.device)
        dones = dones.to(self.device)

        current_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            masked_next_q = torch.full_like(next_q_all, -torch.inf)
            masked_next_q[next_valid_masks] = next_q_all[next_valid_masks]
            has_valid = next_valid_masks.any(dim=1, keepdim=True)
            next_q = masked_next_q.max(dim=1, keepdim=True)[0]
            next_q = torch.where(has_valid, next_q, torch.zeros_like(next_q))
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def store_transition(self, state, action, reward, next_state, next_valid_mask, done):
        self.replay_buffer.push(state, action, reward, next_state, next_valid_mask, done)

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.q_net.state_dict())
