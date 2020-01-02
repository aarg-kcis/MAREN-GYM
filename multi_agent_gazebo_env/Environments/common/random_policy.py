import os
import pdb
import gym
import time
import torch
import numpy as np

from threading import Lock


from ray.rllib.utils.annotations import override
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY


class RandomPolicy(TorchPolicy):
    """
    Random Policy for an Agent.
    """
    @staticmethod
    def size(x):
        from gym.spaces import Discrete, Box
        if isinstance(x, Discrete):
            return x.n
        elif isinstance(x, Box):
            return int(np.prod(x.shape))
        else:
            raise ValueError('This type of space is not supported')


    def __init__(self, observation_space, action_space, config=None):
        self.config = config or {}
        self.lock = Lock()
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.obs_space = self.observation_space = observation_space
        self.act_space = self.action_space = action_space
        # self.obs_input = MBRLPolicy.size(self.obs_space)
        # self.act_input = MBRLPolicy.size(self.act_space)
        # self.obs_space_interval = self.obs_space.high - self.obs_space.low
        # self.obs_space_interval = self.convert(self.obs_space_interval)

    def convert(self, arr):
        tensor = torch.from_numpy(np.asarray(arr))
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor.to(self.device)

    @override(TorchPolicy)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        return_states=False,
                        eval_mode='T',
                        **kwargs):
        # print(obs_batch)
        return ([self.action_space.sample()], [], {})
        # with self.lock:
        #     with torch.no_grad():
        #         z = self.convert(obs_batch)
        #         u = torch.rand(z.size(0), self.act_input)
        # return ([u], [], {}) if return_states else ([u], [], {})

    def get_initial_state(self):
        return []
