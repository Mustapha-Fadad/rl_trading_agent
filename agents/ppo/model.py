# agents/ppo/model.py

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Actor-Critic network with:
      - Shared LSTM encoder for the sequential price/return window
      - Shared scalar linear branch for balance, position_frac, indicators
      - Independent actor head  (policy logits  → action_size)
      - Independent critic head (state value    → 1)

    Separating the actor and critic heads eliminates gradient interference
    between the policy and value objectives.

    State layout expected (produced by TradingEnv._get_state):
        0 .. W-1    : normalised price window   }  (price_t, return_t) per step
        W .. 2W-1   : step-returns window       }  → LSTM
        2W          : normalised balance         \\
        2W+1        : position_frac              |  → scalar branch
        2W+2        : current drawdown            |
        2W+3        : rebalance cooldown progress |
        2W+4 ..     : technical indicators       /
    """

    def __init__(self, state_size, action_size, window_size=10, dropout=0.0):
        super().__init__()

        self.window_size = window_size
        self.seq_size    = window_size * 2
        self.scalar_size = state_size - self.seq_size

        # ---- Shared encoder ----
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.scalar_net = nn.Sequential(
            nn.Linear(self.scalar_size, 32),
            nn.ReLU(),
        )

        combined = 64 + 32

        # ---- Independent actor head ----
        self.actor_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

        # ---- Independent critic head ----
        self.critic_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Initialize the final layer of the actor head so that logits start near 0.
        # This keeps the initial policy perfectly uniform, preventing early collapse.
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.constant_(self.actor_head[-1].bias, 0.0)

    def _encode(self, x):
        seq_part    = x[:, :self.seq_size]
        scalar_part = x[:, self.seq_size:]

        prices  = seq_part[:, :self.window_size]           # (B, W)
        rets    = seq_part[:, self.window_size:]            # (B, W)
        lstm_in = torch.stack([prices, rets], dim=2)       # (B, W, 2)

        lstm_out, _ = self.lstm(lstm_in)                   # (B, W, 128)
        lstm_feat   = lstm_out[:, -1, :]                   # (B, 128)

        scalar_feat = self.scalar_net(scalar_part)         # (B, 64)
        return torch.cat([lstm_feat, scalar_feat], dim=1)  # (B, 192)

    def forward(self, x):
        features = self._encode(x)
        return self.actor_head(features), self.critic_head(features)
