# agents/ppo/memory.py
import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, action_size=5):
        self.action_size = action_size
        self.states            = []
        self.actions           = []
        self.logprobs          = []
        self.rewards           = []
        self.values            = []
        self.dones             = []
        self.valid_action_masks = []

    def __len__(self):
        return len(self.states)

    def add(self, state, action, logprob, reward, value, done, valid_actions):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        # Build full mask of size action_size
        mask = np.zeros(self.action_size, dtype=bool)
        mask[np.array(valid_actions, dtype=int)] = True
        self.valid_action_masks.append(mask)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.valid_action_masks.clear()

    def get_data(self, device):
        states             = torch.FloatTensor(np.array(self.states)).to(device)
        actions            = torch.LongTensor(np.array(self.actions)).to(device)
        logprobs           = torch.FloatTensor(np.array(self.logprobs)).to(device)
        rewards            = torch.FloatTensor(np.array(self.rewards)).to(device)
        values             = torch.FloatTensor(np.array(self.values)).to(device)
        masks              = torch.FloatTensor(1 - np.array(self.dones)).to(device)
        valid_action_masks = torch.BoolTensor(np.array(self.valid_action_masks)).to(device)
        return states, actions, logprobs, rewards, values, masks, valid_action_masks
