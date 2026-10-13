# agents/random_agent.py
"""
Stateless baseline agent that picks uniformly at random from the set of
valid actions at each step.  No training required.
"""
import numpy as np


class RandomAgent:
    """Uniformly random valid-action baseline."""

    def act(self, state, valid_actions):
        """Return a random action from valid_actions (ignores state)."""
        return int(np.random.choice(valid_actions))

    def select_action(self, state, valid_actions=None, deterministic=False):
        """Compatible interface with DQNAgent / PPOAgent for the evaluator."""
        from utils.config import COMMON
        if valid_actions is None:
            valid_actions = list(range(COMMON["action_size"]))
        return self.act(state, valid_actions)
