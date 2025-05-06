# agent.py
#
# Stand-alone replacement for the previous *agent.py*.
# Drop it straight into trading/agent.py and the rest of the
# PICTUR3D code-base will keep working unchanged.

import math
import pickle
import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# --------------------------------------------------------------------------- #
# 1.  Noisy linear layer                                     #
# --------------------------------------------------------------------------- #
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        μ_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-μ_range, μ_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-μ_range, μ_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# --------------------------------------------------------------------------- #
# 2.  Dueling DQN with optional LSTM                                         #
# --------------------------------------------------------------------------- #
class DuelingDQN(nn.Module):
    """
    If `use_lstm` is True, the network expects input of shape (B, T, F)
    and returns `(Q_values, (h, c))`.  When `use_lstm` is False it
    behaves exactly like a feed-forward network and returns just Q_values.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.use_lstm = use_lstm
        self.action_dim = action_dim

        self.fc1 = NoisyLinear(input_dim, 128)
        self.fc2 = NoisyLinear(128, 128)

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=lstm_hidden,
                num_layers=num_layers,
                batch_first=True,
            )
            lstm_out = lstm_hidden
        else:
            lstm_out = 128

        # value & advantage streams
        self.value_fc = NoisyLinear(lstm_out, 64)
        self.value = NoisyLinear(64, 1)
        self.adv_fc = NoisyLinear(lstm_out, 64)
        self.adv = NoisyLinear(64, action_dim)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args
        ----
        x  : (B,F) *or* (B,T,F)
        hc : optional (h,c) from a previous call

        Returns
        -------
        qvals : (B, action_dim)
        hc    : new hidden-state (only if use_lstm=True)
        """
        if x.dim() == 2:  # (B,F)  -> (B,1,F)
            x = x.unsqueeze(1)

        b, t, f = x.shape  # keep for later

        # two dense layers applied time-step wise
        x = x.reshape(b * t, f)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(b, t, -1)

        if self.use_lstm:
            x, hc = self.lstm(x, hc)  # keep running state
        else:
            hc = None

        x = x[:, -1]  # last time-step

        # dueling heads
        val = F.relu(self.value_fc(x))
        val = self.value(val)
        adv = F.relu(self.adv_fc(x))
        adv = self.adv(adv)

        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q, hc

    # convenience – iterate through sub-modules
    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# --------------------------------------------------------------------------- #
# 3.  Prioritised Replay Buffer (unchanged API, but now in one file)          #
# --------------------------------------------------------------------------- #
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = map(
            np.array, zip(*samples)
        )
        return (
            states,
            actions,
            rewards,
            next_states,
            dones.astype(np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio


# --------------------------------------------------------------------------- #
# 4.  TradingAgent                                                            #
# --------------------------------------------------------------------------- #
class TradingAgent:
    """
    Drop-in replacement – public API unchanged.
    Improvements:
    • persistent LSTM hidden state during *inference*
    • resets to zero during mini-batch *training*
    • All NoisyLinear layers re-sample noise each optimiser step
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        use_lstm: bool = True,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        device: str = "cpu",
        buffer_capacity: int = 20_000,
    ) -> None:
        self.device = device
        self.action_dim = action_dim

        # networks
        self.policy_net = DuelingDQN(
            input_dim=input_dim,
            action_dim=action_dim,
            use_lstm=use_lstm,
        ).to(device)
        self.target_net = DuelingDQN(
            input_dim=input_dim,
            action_dim=action_dim,
            use_lstm=use_lstm,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)

        # replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 0.90
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # carry hidden state between timesteps while trading
        self._hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # --------------------------------------------------------------------- #
    # 4.1  Action selection                                                 #
    # --------------------------------------------------------------------- #
    def select_action(self, state: np.ndarray) -> int:
        """
        state can be a 1-D feature vector or a sequence (T,F).
        The method keeps the LSTM hidden state alive across calls,
        mimicking true online inference.
        """
        # ε-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        # shape handling ------------------------------------------------
        st = torch.FloatTensor(state)
        if st.dim() == 1:
            st = st.unsqueeze(0)       # (1,F)
        st = st.unsqueeze(0).to(self.device)  # (B=1,T,F) or (1,1,F)

        with torch.no_grad():
            qvals, self._hc = self.policy_net(st, self._hc)

        action = int(qvals.argmax(dim=1).item())
        return action

    # --------------------------------------------------------------------- #
    # 4.2  Training                                                         #
    # --------------------------------------------------------------------- #
    def train(self, batch_size: int = 64, beta: float = 0.4) -> Optional[float]:
        if len(self.replay_buffer.buffer) < batch_size:
            return None
        
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay_buffer.sample(batch_size, beta)

        # tensors
        states = torch.FloatTensor(states).to(self.device)           # (B,F)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Reset exploration noise  --------------------------------------
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Forward pass (zero hidden state to avoid leakage across samples)
        q_curr, _ = self.policy_net(states, None)
        q_curr = q_curr.gather(1, actions)  # (B,1)

        with torch.no_grad():
            q_next, _ = self.policy_net(next_states, None)
            next_actions = q_next.argmax(dim=1, keepdim=True)

            q_target_next, _ = self.target_net(next_states, None)
            q_target_next = q_target_next.gather(1, next_actions)

            q_target = rewards + (1.0 - dones) * self.gamma * q_target_next

        # TD loss
        loss = (weights * F.mse_loss(q_curr, q_target, reduction="none")).mean()

        # optimize -------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # ε decay --------------------------------------------------------
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # soft target update --------------------------------------------
        for tgt, src in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)

        # update priorities ---------------------------------------------
        td_errors = (q_curr.detach() - q_target).abs().cpu().numpy().squeeze()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)

        return float(loss.item())

    # --------------------------------------------------------------------- #
    # 4.3  Persistence                                                      #
    # --------------------------------------------------------------------- #
    def save(self, path: str = "agent.pth") -> None:
        torch.save(
            {
                "model":   self.policy_net.state_dict(),
                "epsilon": self.epsilon,
                "buffer":  self.replay_buffer,          # pickles fine
            },
            path,
        )

    def load(self, path: str = "agent.pth") -> None:
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.policy_net.load_state_dict(ckpt["model"])
            self.target_net.load_state_dict(ckpt["model"])
            self.epsilon = ckpt.get("epsilon", self.epsilon)
            self.replay_buffer = ckpt.get("buffer", self.replay_buffer)
        else:                                # legacy files (weights only)
            self.policy_net.load_state_dict(ckpt)
            self.target_net.load_state_dict(ckpt)
