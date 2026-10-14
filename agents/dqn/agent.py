# agents/dqn/agent.py

import torch
import torch.optim as optim
import numpy as np

from .model import DQN
from .replay_buffer import ReplayBuffer
from utils.config import COMMON, DQN_CONFIG


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size=None,
        window_size=None,
        config_overrides=None,
    ):

        config = {**DQN_CONFIG, **(config_overrides or {})}

        self.state_size  = state_size
        self.action_size = action_size if action_size is not None else COMMON["action_size"]
        self.window_size = window_size if window_size is not None else COMMON["window_size"]

        self.gamma         = config.get("gamma", COMMON["gamma"])
        self.epsilon       = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min   = config["epsilon_min"]
        self.lr            = config["lr"]
        self.batch_size    = config["batch_size"]
        self.min_replay_size = config["min_replay_size"]
        self.tau           = config["tau"]
        self.use_double    = config["double_dqn"]
        self.dropout       = config["dropout"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(
            state_size, self.action_size,
            window_size=self.window_size, dropout=self.dropout
        ).to(self.device)
        self.target_model = DQN(
            state_size, self.action_size,
            window_size=self.window_size, dropout=0.0
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["lr_decay_step"],
            gamma=config["lr_decay_gamma"],
        )

        self.memory = ReplayBuffer(config["replay_buffer_size"])
        self.optimization_steps = 0
        self.hard_update_target()

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def hard_update_target(self):
        """Copy online network weights to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update_target(self):
        """Polyak averaging: slowly blend online weights into target network."""
        for tp, op in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state, valid_actions=None):
        if np.random.rand() < self.epsilon:
            if valid_actions is not None:
                return int(np.random.choice(valid_actions))
            return np.random.randint(self.action_size)

        self.model.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        self.model.train()

        if valid_actions is not None:
            mask = torch.zeros_like(q_values, dtype=torch.bool)
            mask[:, valid_actions] = True
            q_values = q_values.masked_fill(~mask, float('-inf'))

        return torch.argmax(q_values).item()

    # ------------------------------------------------------------------
    # Valid-action mask reconstruction from batched next-states
    # ------------------------------------------------------------------

    def _valid_action_mask_from_states(self, states):
        """Reconstruct the environment's holding-period mask from observations."""
        mask = torch.ones(
            (states.shape[0], self.action_size),
            dtype=torch.bool,
            device=states.device,
        )
        position_index = self.window_size * 2 + 1
        cooldown_index = self.window_size * 2 + 3
        locked = states[:, cooldown_index] < 1.0 - 1e-6
        if locked.any():
            nearest = torch.clamp(
                torch.round(states[locked, position_index] * (self.action_size - 1)).long(),
                0,
                self.action_size - 1,
            )
            mask[locked] = False
            mask[locked, nearest] = True
        return mask

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        if len(self.memory) < max(self.batch_size, self.min_replay_size):
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_valid_mask = self._valid_action_mask_from_states(next_states)

            if self.use_double:
                # Double DQN: online network selects, target network evaluates
                self.model.eval()
                online_next_q = self.model(next_states)
                self.model.train()
                online_next_q = online_next_q.masked_fill(~next_valid_mask, float('-inf'))
                best_actions  = online_next_q.argmax(dim=1, keepdim=True)

                target_next_q = self.target_model(next_states)
                target_next_q = target_next_q.masked_fill(~next_valid_mask, float('-inf'))
                next_q_values = target_next_q.gather(1, best_actions).squeeze()
            else:
                target_next_q = self.target_model(next_states)
                target_next_q = target_next_q.masked_fill(~next_valid_mask, float('-inf'))
                next_q_values = target_next_q.max(1)[0]

        target = rewards + self.gamma * next_q_values * (1 - dones)
        loss   = torch.nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimization_steps += 1
        self.soft_update_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def step_scheduler(self):
        """Call at end of each episode to decay learning rate."""
        if self.optimization_steps:
            self.scheduler.step()
