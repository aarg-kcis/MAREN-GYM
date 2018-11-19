import os
import sys

import gym
import json
import numpy as np
from mpi4py import MPI

from baselines import logger
# import baselines.her.experiment.config as config
from rollout import RolloutWorker
from ddpg import DDPG
# import ddpg
# print(ddpg, DDPG.__file__)
# sys.exit(0)
import config
# from baselines.her.util import mpi_fork

from subprocess import CalledProcessError
sys.path.append(os.getcwd()+"/../")
sys.path.append('/home/abhay/MAREN-GYM/multi_agent_gazebo_env/Environments/kobuki_simple_world/')
import multi_agent_gazebo_env

env = gym.make("KobukiFormation-v0")
#############################################################################
logger.configure(dir="/tmp/gym-log")
ddpg_params = dict()
params = config.DEFAULT_PARAMS
params["env_name"] = env.env_name


params["T"] = env.max_episode_steps
params["gamma"] = 1. - 1./params["T"]
for name in ['buffer_size', 'hidden', 'layers',
             'network_class',
             'polyak',
             'batch_size', 'Q_lr', 'pi_lr',
             'norm_eps', 'norm_clip', 'max_u',
             'action_l2', 'clip_obs', 'scope', 'relative_goals']:
  ddpg_params[name] = params[name]
  params['_' + name] = params[name]
  del params[name]
params['ddpg_params'] = ddpg_params
params.update(config.DEFAULT_ENV_PARAMS["KobukiFormation-v0"])
with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
  json.dump(params, f)
#############################################################################
dims = config.configure_dims(params)
policy = config.configure_ddpg(dims, params)
#############################################################################

import traceback as tb
try:
  rollout_params = {
          'exploit': False,
          'use_target_net': False,
          'use_demo_states': True,
          'compute_Q': False,
          'T': params['T'],
          'random_eps': 0.3,
          'noise_eps': 0.2

      }
  rollout_worker = RolloutWorker(env.agent_envs, policy, dims, logger, **rollout_params)
  rollout_worker.clear_history()
  for _ in range(params['n_cycles']):
    episode = rollout_worker.generate_rollouts()
  policy.store_episode(episode)

  np.savetxt(logger.get_dir()+"/obs.csv", episode['o'].reshape((-1,10)), delimiter=",")
  np.savetxt(logger.get_dir()+"/ag.csv", episode['ag'].reshape((-1, 4)), delimiter=",")
  np.savetxt(logger.get_dir()+"/goal.csv", episode['g'].reshape((-1, 4)), delimiter=",")
  # print ({k: v.shape for k, v in episode.items()})
except Exception as e:
  print (tb.format_exc())
  print (e)
  print (env.close())
  sys.exit(255)
  
#############################################################################


def train(policy: DDPG, 
          rollout_worker: RolloutWorker,
          n_epochs: int,
          n_test_rollouts: int,
          n_cycles: int,
          n_batches: int):
  latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
  best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
  periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')
  logger.info("Training...")
  best_success_rate = -1
  for epoch in range(n_epochs):
    rollout_worker.clear_history()
    for _ in range(n_cycles):
      episode = rollout_worker.generate_rollouts()
      print (episode)
      sys.exit(0)

  pass