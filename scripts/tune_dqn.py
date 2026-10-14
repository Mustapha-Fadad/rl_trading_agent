import argparse
import json
import os

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from agents.dqn.agent import DQNAgent
from data.data_loader import load_data, train_test_split
from env.trading_env import TradingEnv
from utils.checkpoints import PIPELINE_VERSION
from utils.config import COMMON, DATA_CONFIG, DQN_CONFIG, REWARD_CONFIG, TRAIN_CONFIG
from utils.metrics import buy_and_hold_values, validation_score
from utils.reproducibility import set_seed
from utils.validation import split_validation_windows


def robust_seed_score(seed_scores, stability_weight):
    """Reward mean validation quality while penalizing random-seed sensitivity."""
    scores = np.asarray(seed_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("at least one seed score is required")
    spread = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
    return float(np.mean(scores) - stability_weight * spread)


def prepare_datasets(tickers, val_windows, val_window_size):
    """Build tuning datasets exclusively from each ticker's training split."""
    datasets = []
    for ticker in tickers:
        data = load_data(ticker, **DATA_CONFIG)
        train_data, _ = train_test_split(data, train_ratio=COMMON["train_split"])
        train_only, windows = split_validation_windows(
            train_data,
            observation_window=COMMON["window_size"],
            num_windows=val_windows,
            window_rows=val_window_size,
        )
        datasets.append((ticker, train_only, windows))
        print(
            f"  {ticker}: {len(train_only)} tuning-train rows / "
            f"{len(windows)} x {val_window_size}-day validation windows"
        )
    return datasets


def make_train_envs(datasets):
    return [
        (
            ticker,
            TradingEnv(
                train_data,
                window_size=COMMON["window_size"],
                random_start=TRAIN_CONFIG["random_start"],
                episode_length=TRAIN_CONFIG["episode_length"],
                **REWARD_CONFIG,
            ),
        )
        for ticker, train_data, _ in datasets
    ]


def evaluate_validation_windows(agent, datasets):
    previous_epsilon = agent.epsilon
    was_training = agent.model.training
    agent.epsilon = 0.0
    agent.model.eval()
    scores = []

    try:
        for _, _, windows in datasets:
            for validation_data in windows:
                env = TradingEnv(
                    validation_data,
                    window_size=COMMON["window_size"],
                    random_start=False,
                    **REWARD_CONFIG,
                )
                state = env.reset()
                portfolio_values = [env.balance]
                done = False
                while not done:
                    action = agent.act(state, valid_actions=env.get_valid_actions())
                    state, _, done = env.step(action)
                    portfolio_values.append(env.portfolio_value())

                benchmark_prices = env.data.iloc[
                    env.window_size - 1 :
                ]["Close"].to_numpy(dtype=float)
                benchmark_values = buy_and_hold_values(
                    benchmark_prices,
                    initial_value=portfolio_values[0],
                    transaction_fee=env.transaction_fee,
                )
                scores.append(
                    validation_score(
                        portfolio_values,
                        env.total_turnover,
                        benchmark_values,
                    )
                )
    finally:
        agent.epsilon = previous_epsilon
        agent.model.train(was_training)

    return float(np.mean(scores))


def train_one_seed(datasets, episodes, seed, config_overrides):
    set_seed(seed)
    train_envs = make_train_envs(datasets)
    state_size = train_envs[0][1].reset().shape[0]
    agent = DQNAgent(
        state_size,
        action_size=COMMON["action_size"],
        window_size=COMMON["window_size"],
        config_overrides=config_overrides,
    )

    for episode in range(episodes):
        _, env = train_envs[episode % len(train_envs)]
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, valid_actions=env.get_valid_actions())
            next_state, reward, done = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            agent.train()
            state = next_state
        agent.step_scheduler()

    return evaluate_validation_windows(agent, datasets)


def make_objective(datasets, episodes, seeds, seed_start, stability_weight):
    def objective(trial):
        config_overrides = {
            "lr": trial.suggest_float("lr", 1e-4, 8e-4, log=True),
            "gamma": trial.suggest_float("gamma", 0.97, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
            "epsilon_decay": trial.suggest_float(
                "epsilon_decay", 0.9998, 0.99998
            ),
            "epsilon_min": trial.suggest_float("epsilon_min", 0.03, 0.10),
        }
        seed_scores = []
        print(f"\nTrial {trial.number} | {config_overrides}")

        for seed_offset in range(seeds):
            seed = seed_start + seed_offset
            score = train_one_seed(datasets, episodes, seed, config_overrides)
            seed_scores.append(score)
            partial_score = robust_seed_score(seed_scores, stability_weight)
            trial.report(partial_score, step=seed_offset)
            print(
                f"  seed {seed}: validation={score:+.6f} | "
                f"robust aggregate={partial_score:+.6f}",
                flush=True,
            )
            if trial.should_prune():
                trial.set_user_attr("seed_scores", json.dumps(seed_scores))
                raise optuna.TrialPruned()

        mean_score = float(np.mean(seed_scores))
        score_std = float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0
        trial.set_user_attr("mean_seed_score", mean_score)
        trial.set_user_attr("seed_score_std", score_std)
        trial.set_user_attr("seed_scores", json.dumps(seed_scores))
        return robust_seed_score(seed_scores, stability_weight)

    return objective


def ensure_storage_directory(storage):
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        database_path = storage[len(prefix) :]
        directory = os.path.dirname(database_path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def configure_study(study, args):
    """Prevent resumed studies from mixing trials with different protocols."""
    tuning_config = {
        "pipeline_version": PIPELINE_VERSION,
        "search_space_version": 1,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "tickers": list(args.tickers),
        "val_windows": args.val_windows,
        "val_window_size": args.val_window_size,
        "stability_weight": args.stability_weight,
        "data_config": DATA_CONFIG,
        "reward_config": REWARD_CONFIG,
    }
    existing = study.user_attrs.get("tuning_config")
    if existing is not None and existing != tuning_config:
        raise ValueError(
            "study configuration does not match this command; use a new "
            "--study-name (and preferably a separate --storage file)"
        )
    study.set_user_attr("tuning_config", tuning_config)


def enqueue_default_control(study):
    """Include the current production defaults as a fair tuning baseline."""
    study.enqueue_trial(
        {
            "lr": DQN_CONFIG["lr"],
            "gamma": COMMON["gamma"],
            "tau": DQN_CONFIG["tau"],
            "batch_size": DQN_CONFIG["batch_size"],
            "dropout": DQN_CONFIG["dropout"],
            "epsilon_decay": DQN_CONFIG["epsilon_decay"],
            "epsilon_min": DQN_CONFIG["epsilon_min"],
        },
        user_attrs={"control": "current_defaults"},
        skip_if_exists=True,
    )


def export_trials(study, args):
    trials = study.trials_dataframe()
    protocol = {
        "protocol_pipeline_version": PIPELINE_VERSION,
        "protocol_search_space_version": 1,
        "protocol_study_name": args.study_name,
        "protocol_episodes": args.episodes,
        "protocol_seeds": args.seeds,
        "protocol_seed_start": args.seed_start,
        "protocol_tickers": "+".join(args.tickers),
        "protocol_val_windows": args.val_windows,
        "protocol_val_window_size": args.val_window_size,
        "protocol_stability_weight": args.stability_weight,
    }
    for column, value in reversed(list(protocol.items())):
        trials.insert(0, column, value)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    trials.to_csv(args.output, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed, training-only hyperparameter tuning for DQN"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of additional trials; use 0 to export an existing study",
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=COMMON["seed"])
    parser.add_argument("--tickers", nargs="+", default=TRAIN_CONFIG["tickers"])
    parser.add_argument("--val-windows", type=int, default=TRAIN_CONFIG["val_windows"])
    parser.add_argument(
        "--val-window-size", type=int, default=TRAIN_CONFIG["val_window_size"]
    )
    parser.add_argument("--stability-weight", type=float, default=0.5)
    parser.add_argument("--study-name", default="dqn-v6-multiseed")
    parser.add_argument("--storage", default="sqlite:///logs/dqn_optuna.db")
    parser.add_argument("--output", default="reports/dqn_tuning_trials.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials < 0 or args.episodes < 1 or args.seeds < 1:
        raise ValueError("trials must be non-negative; episodes and seeds must be positive")
    if args.stability_weight < 0:
        raise ValueError("stability-weight must be non-negative")

    ensure_storage_directory(args.storage)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed_start),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    configure_study(study, args)
    if args.trials:
        enqueue_default_control(study)
        print("Preparing DQN tuning data from training splits only...")
        datasets = prepare_datasets(
            args.tickers,
            val_windows=args.val_windows,
            val_window_size=args.val_window_size,
        )
        objective = make_objective(
            datasets,
            episodes=args.episodes,
            seeds=args.seeds,
            seed_start=args.seed_start,
            stability_weight=args.stability_weight,
        )
        study.optimize(objective, n_trials=args.trials, n_jobs=1)
    export_trials(study, args)

    completed_trials = study.get_trials(
        deepcopy=False, states=(TrialState.COMPLETE,)
    )
    print("\n========================================")
    print(f"Completed trials: {len(completed_trials)}")
    if not completed_trials:
        print(f"No completed trials. Empty table saved to {args.output}")
        return
    print(f"Best robust validation score: {study.best_value:+.6f}")
    print(
        "Best seed mean/std: "
        f"{study.best_trial.user_attrs['mean_seed_score']:+.6f} / "
        f"{study.best_trial.user_attrs['seed_score_std']:.6f}"
    )
    print("Best parameters:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value}")
    print(f"Trials saved to {args.output}")
    print("The benchmark and walk-forward test folds were not used for selection.")


if __name__ == "__main__":
    main()
