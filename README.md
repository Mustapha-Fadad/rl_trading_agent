# Deep Reinforcement Learning Trading Agent

An end-to-end research prototype for training and evaluating long-only trading
agents with Deep Q-Learning (DQN) and Proximal Policy Optimization (PPO). The
project covers historical data ingestion, technical features, chronological
evaluation, multi-seed walk-forward validation, an interactive dashboard, and
paper execution through Alpaca.

> This project is for research and education. It is not financial advice and is
> not intended for live-money trading.

## Current Results

Pipeline V6 was evaluated on AAPL using 10 chronological walk-forward folds and
three independent seeds per fold.

| Strategy | Mean return | Mean alpha vs B&H | Max drawdown | Sharpe | Winning folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| DQN | 18.33% | -8.96% | 15.72% | 0.982 | 2/10 |
| PPO | 24.53% | -2.75% | 18.24% | 1.036 | 2/10 |
| SMA-200 | 14.04% | -13.25% | 18.37% | 0.592 | 0/10 |
| Trend + volatility | 16.35% | -10.94% | 14.62% | 0.784 | 2/10 |
| Buy & Hold | 27.29% | - | 20.67% | 1.038 | - |

DQN reduced average drawdown relative to Buy & Hold but sacrificed return and
showed meaningful variation across seeds. PPO frequently converged to 100%
exposure, making it close to Buy & Hold rather than a distinct source of alpha.
In this comparison, DQN produced higher mean return and Sharpe than both simple
rule-based strategies, while trend-plus-volatility produced the lowest average
drawdown. The results are reported honestly: neither RL agent demonstrates
consistent market outperformance.

Detailed outputs are available in:

- `reports/evaluation_summary.csv`
- `reports/walk_forward_results.csv`
- `reports/baseline_walk_forward_results.csv`

## Features

- Target allocations of 0%, 25%, 50%, 75%, or 100% equity exposure
- Transaction fees, turnover penalties, drawdown penalties, and a 20-bar
  minimum holding period
- DQN with replay memory, target network, Double DQN, and action masking
- PPO with an LSTM actor-critic, GAE, clipped updates, and decision-aware action
  masking
- Technical features including SMA, EMA, RSI, MACD, momentum, volatility,
  normalized volume, and rolling price drawdown
- Chronological train/test splits and multi-seed walk-forward validation
- Buy & Hold, cash, and multi-seed random baselines
- SMA-200 trend and trend-plus-volatility rule-based baselines
- Versioned checkpoints that reject incompatible model pipelines
- Streamlit dashboard for evaluation and training reports
- Alpaca paper-order execution with market-hours, cooldown, open-order, and
  duplicate-bar guards

## Project Structure

```text
agents/                 DQN and PPO implementations
data/                   Historical loaders and feature engineering
env/                    Target-allocation trading environment
scripts/                Walk-forward, tuning, and paper-trading commands
training/               Full DQN and PPO training entry points
utils/                  Configuration, metrics, validation, and checkpoints
tests/                  Unit and integration tests
reports/                Reproducible evaluation CSV files
app.py                  Streamlit dashboard
benchmark_evaluation.py Multi-ticker benchmark suite
evaluate_agent.py       Deterministic evaluation command
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Historical data is downloaded with Yahoo Finance and cached locally. Cached
data, trained checkpoints, credentials, and runtime logs are intentionally not
tracked by Git.

## Training

```powershell
python -m training.train_dqn
python -m training.train_ppo
```

Checkpoints are written to `models/`. A checkpoint contains its algorithm,
pipeline version, state size, and window size so incompatible models cannot be
silently loaded.

## Evaluation

Run the deterministic benchmark on AAPL, MSFT, and GOOGL:

```powershell
python benchmark_evaluation.py
```

Run the full AAPL walk-forward experiment used for the published V6 report:

```powershell
python -m scripts.walk_forward --algo both --ticker AAPL --seeds 3
```

This command retrains 60 independent models and can take several hours. A quick
pipeline check is:

```powershell
python -m scripts.walk_forward --algo dqn --ticker AAPL --episodes 20 --seeds 1
```

Limit an experiment to the first fold and keep its output separate from accepted
reports:

```powershell
python -m scripts.walk_forward --algo dqn --ticker AAPL --seeds 3 --max-folds 1 --output reports/dqn_ablation.csv
```

Walk-forward rows are written after every completed seed, so an interrupted run
retains all finished results.

Evaluate the deterministic rule-based baselines without retraining RL models:

```powershell
python -m scripts.walk_forward --algo baselines --ticker AAPL --output reports/baseline_walk_forward_results.csv
```

## Dashboard

```powershell
streamlit run app.py
```

The dashboard reads generated training and evaluation files. It warns when a
report was produced by an older pipeline version.

## Alpaca Paper Trading

Create `.env` from `.env.example` and provide Alpaca paper-account credentials.
Never commit the populated `.env` file.

Inspect one decision without placing an order:

```powershell
python -m scripts.paper_trade --algo dqn --ticker AAPL --once --dry-run
```

Submit one eligible paper-trading decision:

```powershell
python -m scripts.paper_trade --algo dqn --ticker AAPL --once
```

The agent targets a portfolio allocation; it does not use a fixed take-profit
rule. Stopping the process does not liquidate an existing Alpaca position.
Current execution is intended for one ticker strategy per paper account because
multiple processes would share cash and portfolio-level state.

## Tests

```powershell
python -m pytest -q
```

The current V6 suite contains 31 passing tests.

## Known Limitations

- The tested agents do not consistently outperform Buy & Hold.
- PPO often collapses to a passive 100%-invested policy.
- DQN performance varies materially by random seed.
- The feature set is based on daily price, volume, and technical indicators;
  it does not include fundamentals, macroeconomic data, or news.
- Paper orders are not a substitute for testing real execution latency,
  liquidity, slippage, taxes, or operational failure modes.
- Any future tuning should reserve a new untouched holdout period to avoid
  overfitting the existing walk-forward report.

## License

Released under the MIT License. See `LICENSE`.
