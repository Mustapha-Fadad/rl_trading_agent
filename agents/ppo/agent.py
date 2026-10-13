# agents/ppo/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from .model import ActorCritic
from .memory import RolloutBuffer
from utils.config import COMMON, PPO_CONFIG


class PPOAgent:
    def __init__(self, state_size, action_size=None, window_size=None):
        self.state_size  = state_size
        self.action_size = action_size if action_size is not None else COMMON["action_size"]
        self.window_size = window_size if window_size is not None else COMMON["window_size"]

        self.gamma         = COMMON["gamma"]
        self.eps_clip      = PPO_CONFIG["eps_clip"]
        self.gae_lambda    = PPO_CONFIG["gae_lambda"]
        self.k_epochs      = PPO_CONFIG["k_epochs"]
        self.lr            = PPO_CONFIG["lr"]
        self.entropy_coef  = PPO_CONFIG["entropy_coef"]
        self.value_coef    = PPO_CONFIG["value_coef"]
        self.max_grad_norm = PPO_CONFIG["max_grad_norm"]
        self.dropout       = PPO_CONFIG["dropout"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(
            state_size, self.action_size,
            window_size=self.window_size,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.memory   = RolloutBuffer(action_size=self.action_size)
        self.mse_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state, valid_actions=None):
        """Sample an action stochastically (used during training rollouts)."""
        self.policy.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, state_value = self.policy(state_t)

            if valid_actions is not None:
                mask = torch.zeros_like(action_logits, dtype=torch.bool)
                mask[:, valid_actions] = True
                action_logits = action_logits.masked_fill(~mask, float('-inf'))

            dist           = Categorical(logits=action_logits)
            action         = dist.sample()
            action_logprob = dist.log_prob(action)

        self.policy.train()
        return action.item(), action_logprob.item(), state_value.item()

    def select_action(self, state, valid_actions=None, deterministic=False):
        """Select an action, optionally deterministically (used during evaluation)."""
        if not deterministic:
            action, _, _ = self.act(state, valid_actions=valid_actions)
            return action

        self.policy.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, _ = self.policy(state_t)

            if valid_actions is not None:
                mask = torch.zeros_like(action_logits, dtype=torch.bool)
                mask[:, valid_actions] = True
                action_logits = action_logits.masked_fill(~mask, float('-inf'))

            action = torch.argmax(action_logits, dim=1)

        self.policy.train()
        return action.item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _estimate_value(self, state):
        was_training = self.policy.training
        self.policy.eval()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.policy(state_t)
        self.policy.train(was_training)
        return value.squeeze().item()

    def _compute_advantages(self, rewards, values, masks, bootstrap_value):
        advantages = []
        gae = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for i in reversed(range(len(rewards))):
                next_value = bootstrap_value if i == len(rewards) - 1 else values[i + 1]
                delta = rewards[i] + self.gamma * next_value * masks[i] - values[i]
                gae = delta + self.gamma * self.gae_lambda * masks[i] * gae
                advantages.insert(0, gae)
        return torch.stack(advantages)

    def train(self, next_state=None, next_done=True):
        if not self.memory:
            return None

        (
            states,
            actions,
            old_logprobs,
            rewards,
            values,
            masks,
            valid_action_masks,
        ) = self.memory.get_data(self.device)

        # ---- Generalised Advantage Estimation (GAE) ----
        bootstrap_value = 0.0 if next_done else self._estimate_value(next_state)
        advantages = self._compute_advantages(rewards, values, masks, bootstrap_value)
        returns    = advantages + values

        # Normalise advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # ---- K-epoch policy update ----
        self.policy.train()
        for _ in range(self.k_epochs):
            action_logits, state_values = self.policy(states)
            action_logits = action_logits.masked_fill(~valid_action_masks, float('-inf'))
            dist          = Categorical(logits=action_logits)

            logprobs      = dist.log_prob(actions)
            dist_entropy  = dist.entropy()
            state_values  = state_values.squeeze(-1)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Cooldown transitions have exactly one valid action. Their policy
            # gradient is zero, so including them in the mean dilutes the actor
            # update by roughly the holding-period length. The critic still uses
            # every transition because all rewards are informative for value.
            decision_mask = valid_action_masks.sum(dim=1) > 1
            if decision_mask.any():
                policy_loss = -torch.min(surr1, surr2)[decision_mask].mean()
                entropy_bonus = dist_entropy[decision_mask].mean()
            else:
                policy_loss = torch.zeros((), device=self.device)
                entropy_bonus = torch.zeros((), device=self.device)

            value_loss = self.mse_loss(state_values, returns)
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy_bonus
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

        self.memory.clear()
        return float(loss.detach().cpu())
