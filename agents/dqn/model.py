# agents/dqn/model.py

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    DQN with an LSTM encoder for the sequential price/return window and a
    separate linear branch for scalar features (balance, position_frac,
    technical indicators).

    State layout expected (produced by TradingEnv._get_state):
        0 .. W-1    : normalised price window   }  fed as (price_t, return_t)
        W .. 2W-1   : step-returns window       }  sequence into LSTM
        2W          : normalised balance         \\
        2W+1        : position_frac              |  scalar branch
        2W+2        : current drawdown            |
        2W+3        : rebalance cooldown progress |
        2W+4 ..     : technical indicators       /
    """

    def __init__(self, state_size, action_size, window_size=10, dropout=0.0):
        super().__init__()

        self.window_size = window_size
        self.seq_size    = window_size * 2          # price window + returns window
        self.scalar_size = state_size - self.seq_size  # balance + position + indicators

        # LSTM: each timestep t receives (normalised_price_t, return_t)
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)

        # Scalar feature branch
        self.scalar_net = nn.Sequential(
            nn.Linear(self.scalar_size, 32),
            nn.ReLU(),
        )

        combined = 64 + 32
        self.feature_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_head = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, action_size)

    def forward(self, x):
        seq_part    = x[:, :self.seq_size]          # (B, 2W)
        scalar_part = x[:, self.seq_size:]          # (B, scalar_size)

        prices = seq_part[:, :self.window_size]     # (B, W)
        rets   = seq_part[:, self.window_size:]     # (B, W)
        lstm_in = torch.stack([prices, rets], dim=2)  # (B, W, 2)

        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat   = lstm_out[:, -1, :]

        scalar_feat = self.scalar_net(scalar_part)
        features = self.feature_head(torch.cat([lstm_feat, scalar_feat], dim=1))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
