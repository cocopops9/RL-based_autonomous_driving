"""
Double DQN agent with Dueling architecture for the Highway-Env autonomous driving task.

Architecture rationale:
- DQN is the standard baseline for discrete action spaces with continuous state input.
- Double DQN (van Hasselt et al., 2016) decouples action selection from action evaluation
  in the target computation, reducing the well-known overestimation bias of vanilla DQN.
- Dueling architecture (Wang et al., 2016) separates the Q-function into a state-value
  stream V(s) and an advantage stream A(s,a), which improves learning when many actions
  have similar values -- a common situation in highway driving (e.g., IDLE and FASTER
  are often near-equivalent when the lane is clear).

The replay buffer is a standard circular buffer with uniform sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular replay buffer with uniform sampling."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Dueling Q-Network
# ---------------------------------------------------------------------------

class DuelingQNetwork(nn.Module):
    """
    Dueling network that outputs Q(s,a) = V(s) + A(s,a) - mean(A(s,.)).

    The subtraction of mean(A) ensures identifiability: V(s) is forced to
    approximate the true state value rather than being absorbed into A.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)              # (batch, 1)
        advantage = self.advantage_stream(features)       # (batch, action_dim)
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Double DQN agent with:
    - Dueling Q-network
    - Epsilon-greedy exploration with linear decay
    - Soft target network updates (Polyak averaging)
    - Gradient clipping for stability
    """

    def __init__(
        self,
        state_dim=25,
        action_dim=5,
        hidden_dim=256,
        lr=5e-4,
        gamma=0.99,
        buffer_capacity=15000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
        tau=0.005,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # Exploration schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Step counter (for epsilon decay)
        self.total_steps = 0

    # -------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------

    def select_action(self, state, evaluate=False):
        """
        Epsilon-greedy action selection.
        During evaluation, always pick the greedy action (epsilon = 0).
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    # -------------------------------------------------------------------
    # Store transition
    # -------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # -------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------

    def train_step(self):
        """
        Sample a minibatch from the replay buffer and perform one gradient
        step on the Q-network using the Double DQN target:

            y = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)

        Returns the mean loss (for logging), or None if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions actually taken
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # 1. Use the ONLINE network to SELECT the best action in the next state
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            # 2. Use the TARGET network to EVALUATE Q at that action
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # Huber loss (smooth L1) is more robust to outliers than MSE
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft-update the target network
        self._soft_update()

        # Decay epsilon
        self._decay_epsilon()

        self.total_steps += 1

        return loss.item()

    # -------------------------------------------------------------------
    # Target network update
    # -------------------------------------------------------------------

    def _soft_update(self):
        """Polyak averaging: theta_target <- tau * theta_online + (1-tau) * theta_target."""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    # -------------------------------------------------------------------
    # Epsilon decay
    # -------------------------------------------------------------------

    def _decay_epsilon(self):
        """Linear decay from epsilon_start to epsilon_end over epsilon_decay_steps."""
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    # -------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------

    def save(self, path):
        """Save model weights and training state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path):
        """Load model weights and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]
        self.q_network.eval()
