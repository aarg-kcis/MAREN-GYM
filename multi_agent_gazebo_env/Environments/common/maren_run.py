import os
import pdb
import gym
import time
import pathlib
import logging
import argparse

import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation import RolloutWorker, SampleBatch
from ray.rllib.optimizers.replay_buffer import ReplayBuffer

from gazebo_env import GazeboEnv
from random_policy import RandomPolicy
from gazebo_tb_agent import GazeboTurtleBotAgent

def get_env_maker(env_cls):
    def env_maker(config, return_agents=False):
        env = env_cls(config)
        env = BaseEnv.to_base_env(env)
        if return_agents:
            return env, env.envs[0].agents
        return env
    return env_maker


def training_workflow(config, reporter):
    from gym.spaces import Box
    import numpy as np
    env_maker = get_env_maker(GazeboEnv)
    env, agents = env_maker(config['env_config'], return_agents=True)
    space = Box(low=-np.ones(2), high=np.ones(2))
    # pdb.set_trace()
    replay_buffers = {
        agent_id: ReplayBuffer(config.get('buffer_size', 1000))
        for agent_id in agents
    }
    policy = {
        k: (RandomPolicy, a.observation_space, a.action_space, {})
        for k, a in agents.items()
    }
    worker = RolloutWorker(
        lambda x: env,
        policy=policy,
        batch_steps=32,
        policy_mapping_fn=lambda x: x,
        episode_horizon=20

    )
    for i in range(config['num_iters']):
        T1 = SampleBatch.concat_samples([worker.sample()])
        for agent_id, batch in T1.policy_batches.items():
            for row in batch.rows():
                replay_buffers[agent_id].add(
                    row['obs'],
                    row['actions'],
                    row['rewards'],
                    row['new_obs'],
                    row['dones'], weight=None
                )
    pdb.set_trace()

if __name__ == '__main__':
    from utils import get_configuration
    import rospy
    config = get_configuration('../tb_simple_world')
    config['agents']['waffle']['agent_class'] = GazeboTurtleBotAgent
    # rosrate = rospy.Rate(1) # 10hz
    # config['rr'] = rosrate
    ray.init(logging_level=logging.DEBUG)

    tune.run(
        training_workflow,
        resources_per_trial={
            "gpu": 0,
            "cpu": 1,
            "extra_cpu": 1,
        },
        config={
            "num_workers": 1,
            "num_iters": 20,
            'num_eval_eps': 10,
            'env_config': config
        },
        verbose=2,
        loggers=None,
    )