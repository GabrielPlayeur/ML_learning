import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ("u", "d", "l", "r")

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, n):
        self.buffer.append((state, action, reward, next_state, done, n))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Module:
    def __init__(
        self,
        network: type[nn.Module],
        device: torch.device,
        input_size: int = 8,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay_steps: int = 100_000,
        gamma: float = 0.9,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 1000,
        train_freq: int = 4,
        n_step: int = 3,
    ):
        self.device = device
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # per-step factor so that epsilon goes from its start value to epsilon_min in epsilon_decay_steps steps
        if epsilon > epsilon_min:
            self.epsilon_decay = (epsilon_min / epsilon) ** (1 / epsilon_decay_steps)
        else:
            self.epsilon_decay = 1.0
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq

        self.state_type = getattr(network, "state_type", "vector")
        self.network = network(input_size, len(ACTIONS)).to(self.device)
        self.target_network = network(input_size, len(ACTIONS)).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self._learn_steps = 0
        self._env_steps = 0
        self.last_action_random = False # True if the last action came from the epsilon exploration
        self.last_q = None # mean Q(s, a) of the last batch, kept as a tensor to avoid a GPU sync

        self.n_step = n_step
        self.window = deque() # the last n_step single-step transitions of the current episode

    def action(self, state: np.ndarray) -> str:
        self.last_action_random = random.random() < self.epsilon
        if self.last_action_random:
            return random.choice(ACTIONS)
        with torch.no_grad():
            q_values = self.network(torch.from_numpy(state).unsqueeze(0).to(self.device)).squeeze(0)
        return ACTIONS[int(torch.argmax(q_values))]

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _push_window(self):
        """Aggregate the window into one n-step transition: R = r_0 + g*r_1 + ... and s' / done from the last one."""
        rewards = [t[2] for t in self.window]
        n_reward = sum(self.gamma ** i * r for i, r in enumerate(rewards))
        state, action = self.window[0][:2]
        next_state, done = self.window[-1][3:]
        self.replay_buffer.push(state, action, n_reward, next_state, done, len(self.window))
        self.window.popleft()

    def end_episode(self):
        """Flush the remaining shortened transitions (called at the end of every episode, e.g. when truncated)."""
        while self.window:
            self._push_window()

    def learn(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, done: bool):
        self.window.append((state, action, reward, next_state, done))
        if len(self.window) == self.n_step:
            self._push_window()
        if done:
            self.end_episode()

        self._env_steps += 1
        if len(self.replay_buffer) < self.batch_size or self._env_steps % self.train_freq != 0:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, ns = zip(*batch)

        states_t = torch.from_numpy(np.stack(states)).to(self.device)
        next_states_t = torch.from_numpy(np.stack(next_states)).to(self.device)
        actions_idx = torch.tensor([ACTIONS.index(a) for a in actions], device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        not_done_t = torch.tensor([not d for d in dones], dtype=torch.float32, device=self.device)
        gammas_t = torch.tensor([self.gamma ** n for n in ns], dtype=torch.float32, device=self.device)

        q_values = self.network(states_t).gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        self.last_q = q_values.detach().mean()
        with torch.no_grad():
            best = self.network(next_states_t).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states_t).gather(1, best).squeeze(1)
            targets = rewards_t + gammas_t * next_q_values * not_done_t

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.detach()

    def save(self, path: str | Path):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str | Path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
