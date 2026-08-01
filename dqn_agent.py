"""
Double DQN agent with Dueling architecture for Highway-Env autonomous driving.

Key design decisions:
- Double DQN target: decouples action selection (online net) from evaluation
  (target net), reducing overestimation bias (van Hasselt et al., 2016).
- Dueling architecture: decomposes Q(s,a) = V(s) + A(s,a) - mean(A), which
  helps when many actions are near-equivalent (Wang et al., 2016).
- Hard target updates every N steps for training stability.
- Orthogonal weight initialization for better gradient flow at early training.
- Warmup period: no gradient updates until the buffer has enough diverse samples.
"""

import os
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
    Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
    Mean subtraction ensures identifiability of V and A.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """Double DQN agent with Dueling architecture."""

    def __init__(
        self,
        state_dim=25,
        action_dim=5,
        hidden_dim=128,
        lr=5e-4,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=40000,
        target_update_freq=200,
        learning_starts=1000,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.learning_starts:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1

        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self._decay_epsilon()
        return loss.item()

    def _decay_epsilon(self):
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def save(self, path):
        """Save the model for evaluation: online and target network weights,
        epsilon, step counters and the optimizer state. The replay buffer is not
        included; the full resumable checkpoint is written by
        save_checkpoint()."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path):
        """Load a model saved by save(): network weights plus target network,
        epsilon and step counters. Used for evaluation."""
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

    def save_checkpoint(self, path):
        """
        Save full training state for resumption: weights, optimizer,
        replay buffer contents, exploration schedule, step counter, and
        RNG states for Python random, NumPy, and PyTorch.

        The write is ATOMIC: data is written to a temporary file first and
        then renamed over the target, so a process kill mid-write (e.g.
        Colab session termination) never corrupts an existing checkpoint.
        """
        # Convert deque of tuples to arrays for efficient serialization
        buffer_list = list(self.replay_buffer.buffer)
        if buffer_list:
            states, actions, rewards, next_states, dones = zip(*buffer_list)
            buffer_data = {
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32),
                "next_states": np.array(next_states, dtype=np.float32),
                "dones": np.array(dones, dtype=np.float32),
            }
        else:
            buffer_data = None

        payload = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "buffer_data": buffer_data,
            "buffer_capacity": self.replay_buffer.buffer.maxlen,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state().detach().cpu().to(torch.uint8).contiguous(),
            "torch_cuda_rng": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
        }

        # Atomic write: temp file in the same directory, then rename.
        # Durable, rotated write:
        # 1) write to tmp and fsync (forces DriveFS to upload the bytes,
        #    not just cache them -- a Colab VM kill otherwise loses the write)
        # 2) rotate the current checkpoint to .prev
        # 3) rename tmp over the target
        # A kill at any point leaves at least one complete checkpoint among
        # {path, path.tmp, path.prev}; the loader tries them in that order.
        tmp_path = path + ".tmp"
        prev_path = path + ".prev"
        torch.save(payload, tmp_path)
        fd = os.open(tmp_path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if os.path.exists(path):
            os.replace(path, prev_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path):
        """
        Load full training state. Restores replay buffer and RNG states
        so that training resumes as close as possible to where it stopped.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]

        # Keep q_network in train mode for continued training
        self.q_network.train()
        self.target_network.eval()

        # Restore replay buffer
        buffer_data = checkpoint.get("buffer_data")
        if buffer_data is not None:
            self.replay_buffer = ReplayBuffer(checkpoint["buffer_capacity"])
            n = len(buffer_data["states"])
            for i in range(n):
                self.replay_buffer.push(
                    buffer_data["states"][i],
                    int(buffer_data["actions"][i]),
                    float(buffer_data["rewards"][i]),
                    buffer_data["next_states"][i],
                    float(buffer_data["dones"][i]),
                )

        # Restore RNG states. These only affect the reproducibility of future
        # random draws, so a failure here (e.g. a GPU-saved torch RNG tensor
        # that a different build rejects on set_rng_state) must NOT abort the
        # resume: the network weights, optimizer and replay buffer are already
        # restored above. Each RNG stream is restored independently.
        def _try(label, fn):
            try:
                fn()
            except Exception as e:
                print(f"  note: could not restore {label} RNG state ({e}); "
                      f"continuing without it.")

        if "python_rng" in checkpoint:
            _try("Python", lambda: random.setstate(checkpoint["python_rng"]))
        if "numpy_rng" in checkpoint:
            _try("NumPy", lambda: np.random.set_state(checkpoint["numpy_rng"]))
        if "torch_rng" in checkpoint:
            def _set_torch_rng():
                st = checkpoint["torch_rng"]
                # set_rng_state requires a CPU uint8 ByteTensor; coerce it,
                # rebuilding from raw bytes to guarantee the exact type.
                st = torch.ByteTensor(
                    st.detach().cpu().to(torch.uint8).contiguous().numpy()
                )
                torch.set_rng_state(st)
            _try("PyTorch", _set_torch_rng)
        if checkpoint.get("torch_cuda_rng") is not None and torch.cuda.is_available():
            _try("CUDA", lambda: torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng"]))
