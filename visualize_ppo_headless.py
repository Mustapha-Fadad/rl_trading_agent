# visualize_ppo_headless.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.data_loader import load_data, train_test_split
from env.trading_env import TradingEnv
from agents.ppo.agent import PPOAgent
from utils.config import COMMON, DATA_CONFIG, REWARD_CONFIG
from utils.checkpoints import load_checkpoint

# Load data — evaluate on TEST split
data = load_data("AAPL", **DATA_CONFIG)
_, test_data = train_test_split(data, train_ratio=COMMON["train_split"])

reward_kwargs = REWARD_CONFIG.copy()
env = TradingEnv(test_data, window_size=COMMON["window_size"], **reward_kwargs)

state_size  = env.reset().shape[0]
action_size = COMMON["action_size"]
window_size = COMMON["window_size"]

# Load trained agent
agent = PPOAgent(state_size, action_size=action_size, window_size=window_size)
agent.policy.load_state_dict(load_checkpoint("models/ppo.pth", agent.device, "ppo"))
agent.policy.eval()

state = env.reset()

prices        = []
buy_half_pts  = []
buy_all_pts   = []
sell_half_pts = []
sell_all_pts  = []
portfolio_values = []

done = False

while not done:
    action = agent.select_action(
        state,
        valid_actions=env.get_valid_actions(),
        deterministic=True,
    )
    next_state, reward, done = env.step(action)

    price = float(env.data.iloc[env.current_step - 1]['Close'])
    prices.append(price)

    portfolio_value = env.balance + env.shares_held * price
    portfolio_values.append(portfolio_value)

    idx = len(prices) - 1
    if action == 1:
        buy_half_pts.append((idx, price))
    elif action == 2:
        buy_all_pts.append((idx, price))
    elif action == 3:
        sell_half_pts.append((idx, price))
    elif action == 4:
        sell_all_pts.append((idx, price))

    state = next_state

# ----------- PLOT 1: PRICE + ACTIONS -----------
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(prices, label="Price", color="steelblue", linewidth=1)

if buy_half_pts:
    bx, by = zip(*buy_half_pts)
    ax.scatter(bx, by, marker="^", color="lime",  s=60, label="BUY ½",  zorder=3)
if buy_all_pts:
    bx, by = zip(*buy_all_pts)
    ax.scatter(bx, by, marker="^", color="green", s=90, label="BUY ALL", zorder=3)
if sell_half_pts:
    sx, sy = zip(*sell_half_pts)
    ax.scatter(sx, sy, marker="v", color="orange", s=60, label="SELL ½",  zorder=3)
if sell_all_pts:
    sx, sy = zip(*sell_all_pts)
    ax.scatter(sx, sy, marker="v", color="red",    s=90, label="SELL ALL", zorder=3)

ax.set_title("PPO Trading Agent Decisions (Test Data)")
ax.legend()
plt.tight_layout()
plt.savefig("plot1_ppo.png", dpi=150)
plt.close()

# ----------- PLOT 2: PORTFOLIO VALUE -----------
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(portfolio_values, label="Portfolio Value", color="darkorange")
ax.axhline(portfolio_values[0], color="grey", linestyle="--", linewidth=0.8, label="Initial Value")
ax.set_title("PPO Portfolio Value Over Time (Test Data)")
ax.legend()
plt.tight_layout()
plt.savefig("plot2_ppo.png", dpi=150)
plt.close()

print("Saved plot1_ppo.png and plot2_ppo.png")
