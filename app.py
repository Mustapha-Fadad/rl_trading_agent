# app.py — Streamlit Interactive Dashboard for RL Trading Agent

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.data_loader import load_data, train_test_split
from evaluate_agent import evaluate_agent
from utils.checkpoints import PIPELINE_VERSION
from utils.config import DATA_CONFIG, TRAIN_CONFIG

# Page Config
st.set_page_config(
    page_title="RL Trading Agent Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Subtitle
st.title("📈 Reinforcement Learning Trading Agent")
st.markdown("### Deep Q-Learning (DQN) & Proximal Policy Optimization (PPO) Analytics Dashboard")

# Sidebar
st.sidebar.header(" Control Panel")
ticker = st.sidebar.selectbox("Select Stock Ticker", options=TRAIN_CONFIG["tickers"], index=0)
algo = st.sidebar.radio("Select RL Algorithm", options=["dqn", "ppo"], index=1).lower()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Quick Commands")
st.sidebar.code(f"python -m training.train_{algo}", language="bash")
st.sidebar.code(f"python -m scripts.walk_forward --algo {algo}", language="bash")
st.sidebar.code(f"python -m scripts.paper_trade --algo {algo}", language="bash")

# Helper to run evaluation if model exists
def get_eval_results(algo_name, ticker_symbol):
    model_path = f"models/{algo_name}.pth"
    if not os.path.exists(model_path):
        return None
    try:
        return evaluate_agent(algo=algo_name, model_path=model_path, ticker=ticker_symbol, split="test")
    except Exception as e:
        return None

# Tabs Layout
tab_benchmark, tab_walk_forward, tab_training, tab_paper = st.tabs([
    "Agent Benchmark & Trades",
    "Walk-Forward Validation",
    "Training Curves",
    "Alpaca Live Trading"
])

# -------------------------------------------------------------------
# TAB 1: Agent Benchmark & Interactive Trades
# -------------------------------------------------------------------
with tab_benchmark:
    st.subheader(f"Evaluation Overview — {algo.upper()} on {ticker} (Test Set)")

    # Load dataset
    data = load_data(ticker, **DATA_CONFIG)
    _, test_data = train_test_split(data, train_ratio=0.8)

    # Determine date axis for Plotly
    x_axis = test_data.index if not isinstance(test_data.index, pd.RangeIndex) else test_data.get('Date', test_data.index)

    eval_res = get_eval_results(algo, ticker)

    if eval_res is None:
        st.info(f"No trained checkpoint found at `models/{algo}.pth`. Run `python -m training.train_{algo}` to train the model.")
    else:
        # Metrics Display
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Agent Return", f"{eval_res['agent_return_pct']}%", delta=f"{eval_res['agent_return_pct'] - eval_res['buy_and_hold_return_pct']:.2f}% vs B&H")
        col2.metric("Buy & Hold Return", f"{eval_res['buy_and_hold_return_pct']}%")
        col3.metric(
            "Max Drawdown",
            f"{eval_res['max_drawdown_pct']}%",
            delta=f"{eval_res['buy_and_hold_max_drawdown_pct'] - eval_res['max_drawdown_pct']:.2f}% vs B&H",
        )
        col4.metric(
            "Sharpe",
            f"{eval_res['sharpe_ratio']}",
            delta=f"{eval_res['sharpe_ratio'] - eval_res['buy_and_hold_sharpe_ratio']:.3f} vs B&H",
        )
        col5.metric("Turnover", f"{eval_res['total_turnover']}x")
        col6.metric("Avg Exposure", f"{eval_res['average_exposure_pct']}%")

        st.markdown("---")

    # Interactive Price & Indicators Plotly Chart
    st.markdown("#### Price Action & Technical Indicators (Test Split)")
    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=x_axis, y=test_data['Close'],
        mode='lines', name='Close Price', line=dict(color='#17becf', width=2)
    ))

    if 'SMA_10' in test_data.columns:
        fig_price.add_trace(go.Scatter(
            x=x_axis, y=test_data['SMA_10'],
            mode='lines', name='SMA 10', line=dict(color='#ff7f0e', width=1.5, dash='dash')
        ))
    if 'SMA_30' in test_data.columns:
        fig_price.add_trace(go.Scatter(
            x=x_axis, y=test_data['SMA_30'],
            mode='lines', name='SMA 30', line=dict(color='#2ca02c', width=1.5, dash='dot')
        ))

    fig_price.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Time / Trading Days",
        yaxis_title="Stock Price ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Summary Table if file exists
    if os.path.exists("reports/evaluation_summary.csv"):
        st.markdown("#### Full Multi-Ticker Benchmark Summary")
        df_summary = pd.read_csv("reports/evaluation_summary.csv")
        if "pipeline_version" not in df_summary or not (df_summary["pipeline_version"] == PIPELINE_VERSION).all():
            st.warning("This report predates the current environment. Regenerate it before comparing models.")
        elif "checkpoint_pipeline_version" in df_summary and not (
            df_summary["checkpoint_pipeline_version"].astype(str) == str(PIPELINE_VERSION)
        ).all():
            st.warning("This report contains legacy checkpoints. Retrain before comparing models.")
        st.dataframe(df_summary, use_container_width=True)


# -------------------------------------------------------------------
# TAB 2: Walk-Forward Validation
# -------------------------------------------------------------------
with tab_walk_forward:
    st.subheader("Rolling Walk-Forward Validation Results")
    st.markdown("Evaluates policy robustness across non-overlapping, sliding market regimes.")

    wf_file = "reports/walk_forward_results.csv"
    if not os.path.exists(wf_file):
        st.info("💡 No walk-forward results found yet. Run `python -m scripts.walk_forward --algo both` to generate report!")
    else:
        df_wf = pd.read_csv(wf_file)
        if "pipeline_version" not in df_wf or not (df_wf["pipeline_version"] == PIPELINE_VERSION).all():
            st.warning("These walk-forward results predate the current environment. Regenerate them before use.")

        # Filter algorithm
        sub_wf = df_wf[df_wf["algo"] == algo] if "algo" in df_wf.columns else df_wf

        if sub_wf.empty:
            st.warning(f"No walk-forward data recorded for {algo.upper()}.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            mean_agent = sub_wf["agent_return_pct"].mean()
            mean_bh = sub_wf["bh_return_pct"].mean()
            mean_alpha = sub_wf["alpha_pct"].mean()
            win_folds = (sub_wf["alpha_pct"] > 0).sum()

            col1.metric("Mean Agent Return", f"{mean_agent:.2f}%")
            col2.metric("Mean B&H Return", f"{mean_bh:.2f}%")
            col3.metric("Mean Outperformance (Alpha)", f"+{mean_alpha:.2f}%")
            col4.metric("Outperformed Folds", f"{win_folds} / {len(sub_wf)}")

            st.markdown("---")

            # Per-fold comparison bar chart
            st.markdown("#### Return Comparison per Walk-Forward Fold")
            fig_wf = go.Figure()
            fig_wf.add_trace(go.Bar(
                x=[f"Fold {f}" for f in sub_wf["fold"]],
                y=sub_wf["agent_return_pct"],
                name=f"{algo.upper()} Return",
                marker_color="#00cc96"
            ))
            fig_wf.add_trace(go.Bar(
                x=[f"Fold {f}" for f in sub_wf["fold"]],
                y=sub_wf["bh_return_pct"],
                name="Buy & Hold Return",
                marker_color="#ef553b"
            ))
            fig_wf.update_layout(
                barmode='group',
                template="plotly_dark",
                height=400,
                xaxis_title="Walk-Forward Folds",
                yaxis_title="Return (%)"
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            st.markdown("#### Fold Detailed Table")
            st.dataframe(sub_wf, use_container_width=True)


# -------------------------------------------------------------------
# TAB 3: Training Curves & Convergence
# -------------------------------------------------------------------
with tab_training:
    st.subheader(f"Training Progress & Rewards ({algo.upper()})")

    log_path = f"logs/{algo}_training.csv"
    if not os.path.exists(log_path):
        st.info(f"💡 No training log found at `{log_path}`. Run `python -m training.train_{algo}` to train!")
    else:
        df_log = pd.read_csv(log_path)
        if "pipeline_version" not in df_log or not (df_log["pipeline_version"] == PIPELINE_VERSION).all():
            st.warning("This training log predates the current environment. Retrain before interpreting it.")

        st.markdown("#### Episode Rewards & Moving Average")
        window_smooth = st.slider("Smoothing Window Size (Episodes)", min_value=1, max_value=50, value=20)
        df_log['reward_smooth'] = df_log['reward'].rolling(window=window_smooth, min_periods=1).mean()

        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(
            x=df_log['episode'], y=df_log['reward'],
            mode='lines', name='Episode Reward', opacity=0.3, line=dict(color='#ab63fa')
        ))
        fig_train.add_trace(go.Scatter(
            x=df_log['episode'], y=df_log['reward_smooth'],
            mode='lines', name=f'{window_smooth}-Ep Moving Avg', line=dict(color='#ffa15a', width=2.5)
        ))

        fig_train.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Episode",
            yaxis_title="Total Episode Reward",
            hovermode="x unified"
        )
        st.plotly_chart(fig_train, use_container_width=True)


# -------------------------------------------------------------------
# TAB 4: Alpaca Live Paper Trading Stream
# -------------------------------------------------------------------
with tab_paper:
    st.subheader("⚡ Alpaca Live Paper Trading Stream")
    st.markdown("Real-time order execution and balance monitoring connected to Alpaca API.")

    paper_log = "logs/paper_trading_v6.csv"
    if not os.path.exists(paper_log):
        st.info("💡 No live paper trading logs recorded yet. Run `python -m scripts.paper_trade --algo ppo --ticker AAPL` to start!")
    else:
        df_paper = pd.read_csv(paper_log)
        st.markdown("#### Live Portfolio Value Tracking")
        fig_paper = px.line(
            df_paper, x="timestamp", y="portfolio_value",
            title="Alpaca Account Portfolio Value ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_paper, use_container_width=True)

        st.markdown("#### Recent Live Order Execution Log")
        st.dataframe(df_paper.tail(20), use_container_width=True)
