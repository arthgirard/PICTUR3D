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

import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, use_lstm: bool = False):
        super(DuelingDQN, self).__init__()
        self.use_lstm = use_lstm
        self.fc1 = NoisyLinear(input_dim, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.dropout = nn.Dropout(0.2)
        lstm_output_dim = 128
        if self.use_lstm:
            self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.value_fc = NoisyLinear(lstm_output_dim, 64)
        self.value = NoisyLinear(64, 1)
        self.advantage_fc = NoisyLinear(lstm_output_dim, 64)
        self.advantage = NoisyLinear(64, action_dim)

    def forward(self, x):
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

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.value_fc.reset_noise()
        self.value.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage.reset_noise()

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
    def __init__(self, input_dim: int, action_dim: int, use_lstm: bool = True, 
                 lr: float = 0.0001,
                 gamma: float = 0.95, tau: float = 0.005, device: str = "cpu", buffer_capacity: int = 1000) -> None:
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 0.90
        self.epsilon_min = 0.01
        # Use a multiplicative decay factor (e.g., 0.995 per training step)
        self.epsilon_decay_rate = 0.995

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

    def train(self, batch_size=64, beta=0.4):
        if len(self.replay_buffer.buffer) < batch_size:
            return None
    
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(batch_size, beta)
    
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
    
        # Reset noise for both networks
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
    
        # Compute current Q-values
        current_q = self.policy_net(states).gather(1, actions)
    
        # Compute next Q-values (Double DQN style)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
    
        # Compute loss (with importance sampling weights)
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
    
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
        # Update priorities
        td_errors = (current_q - target_q).detach().cpu().numpy().squeeze()
        new_priorities = np.abs(td_errors) + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)
    
        # Soft update of target network
        self._soft_update_target()
    
        return loss.item()
    

    def _soft_update_target(self) -> None:
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def save(self, path: str = "agent.pth") -> None:
        try:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, path)
            logging.info("Agent saved.")
        except Exception as e:
            logging.error(f"Error saving agent: {e}")

    def load(self, path: str = "agent.pth") -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            logging.info("Agent loaded.")
        except Exception as e:
            logging.error(f"Error loading agent: {e}")
