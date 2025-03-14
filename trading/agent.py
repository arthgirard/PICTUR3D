# agent.py

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from torch.nn.utils import clip_grad_norm_
from typing import Tuple, List, Any, Optional

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, use_lstm: bool = False) -> None:
        super(DuelingDQN, self).__init__()
        self.use_lstm = use_lstm
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.2)
        lstm_output_dim = 128
        if self.use_lstm:
            self.lstm = nn.LSTM(128, 128, batch_first=True)
            lstm_output_dim = 128
        self.value_fc = nn.Linear(lstm_output_dim, 64)
        self.value = nn.Linear(64, 1)
        self.advantage_fc = nn.Linear(lstm_output_dim, 64)
        self.advantage = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if self.use_lstm:
            x = x.unsqueeze(1)
            x, _ = self.lstm(x)
            x = x.squeeze(1)
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
       if len(self.buffer) == self.capacity:
           priorities = self.priorities
       else:
           priorities = self.priorities[:len(self.buffer)]
       probs = priorities ** self.alpha
       probs_sum = probs.sum()
       # Check for NaN or zero sum and fallback to uniform probabilities
       if np.isnan(probs_sum) or probs_sum == 0:
           probs = np.ones_like(probs) / len(probs)
       else:
           probs /= probs_sum
       indices = np.random.choice(len(self.buffer), batch_size, p=probs)
       samples = [self.buffer[idx] for idx in indices]
       total = len(self.buffer)
       weights = (total * probs[indices]) ** (-beta)
       weights /= weights.max()
       weights = np.array(weights, dtype=np.float32)
       batch = list(zip(*samples))
       states = np.array(batch[0])
       actions = np.array(batch[1])
       rewards = np.array(batch[2])
       next_states = np.array(batch[3])
       dones = np.array(batch[4])
       return states, actions, rewards, next_states, dones, indices, weights


    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def save(self, path: str) -> None:
        try:
            with open(path, 'wb') as f:
                pickle.dump((self.buffer, self.pos, self.priorities), f)
            logging.info(f"Replay buffer saved to {path}")
        except Exception as e:
            logging.error(f"Error saving replay buffer: {e}")

    def load(self, path: str) -> None:
        try:
            with open(path, 'rb') as f:
                self.buffer, self.pos, self.priorities = pickle.load(f)
            logging.info(f"Replay buffer loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading replay buffer: {e}")


class TradingAgent:
    def __init__(self, input_dim: int, action_dim: int, use_lstm: bool = False, lr: float = 1e-4, gamma: float = 0.99, tau: float = 0.005, device: str = "cpu", buffer_capacity: int = 10000) -> None:
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.policy_net = DuelingDQN(input_dim, action_dim, use_lstm).to(self.device)
        self.target_net = DuelingDQN(input_dim, action_dim, use_lstm).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
            logging.debug(f"Random action: {action}")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()
            logging.debug(f"Policy action: {action}")
            return action

    def train(self, batch_size: int = 64, beta: float = 0.4) -> Optional[float]:
        if len(self.replay_buffer.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(batch_size, beta)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = (weights * nn.functional.mse_loss(current_q, target_q, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        td_errors = (current_q - target_q).detach().cpu().numpy().squeeze()
        new_priorities = np.abs(td_errors) + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)
        self._soft_update_target()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

    def _soft_update_target(self) -> None:
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def save(self, path: str = "agent.pth", buffer_path: str = "replay_buffer.pkl") -> None:
        try:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, path)
            self.replay_buffer.save(buffer_path)
            logging.info("Agent and replay buffer saved.")
        except Exception as e:
            logging.error(f"Error saving agent: {e}")

    def load(self, path: str = "agent.pth", buffer_path: str = "replay_buffer.pkl") -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            if os.path.exists(buffer_path):
                self.replay_buffer.load(buffer_path)
            logging.info("Agent and replay buffer loaded.")
        except Exception as e:
            logging.error(f"Error loading agent: {e}")
