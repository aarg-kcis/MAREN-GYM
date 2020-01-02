import os
import sys
import gym
import yaml
import time
import enum
import rospy
import numpy as np

from utils import check_for_ROS
from rospy import Duration, Time
from tf import TransformListener
from gazebo_env import GazeboAgent
from preprocessor import Preprocessor
from tf.transformations import euler_from_quaternion as q2e

class GazeboTurtleBotAgent(GazeboAgent):
    def __init__(self, agent_type, _id, config):
        super().__init__(agent_type, _id, config)
        self.tf_obs = {
            '{}_{}'.format(self.type, i+1): None
            for i in range(config['num_agents']) if i+1 != self.id
        }
        ppd = {'laserscan': GazeboTurtleBotAgent.laserscan_preprocessor}
        ppd.update(
            dict.fromkeys(self.tf_obs, GazeboTurtleBotAgent.tf_preprocessor))
        self.preprocessor = Preprocessor(ppd)
        self._is_obs_ready.update(dict.fromkeys(self.tf_obs, False))
        tf_space = gym.spaces.Box(
            low=np.array([-10, -10, -np.pi]),
            high=np.array([10, 10, np.pi]), dtype=np.float32
        )
        self.observation_space.spaces.update(
            dict.fromkeys(self.tf_obs, tf_space)
        )
        # check_for_ROS()
        print('='*100)
        print(self)
        print(rospy.get_name())
        print(rospy.get_namespace())
        print(rospy.get_node_uri())
        print('='*100)

    def reset(self):
        super().reset()
        self.tfl = TransformListener(True, Duration(5))


    def get_obs(self, apply_preprocessor=True):
        t = time.time()
        print(self, 'GET_OBS')
        obs = super().get_obs(apply_preprocessor=False)
        print('GOT OTHER OBS, WAITING FOR TF ...')
        src = '/{}/base_link'.format(self.ns)
        nbhrs = list(self.tf_obs.keys())
        while nbhrs:
            nbhr = nbhrs[-1]
            try:
                tgt = '/{}/base_link'.format(nbhr)
                self.tf_obs[nbhr] = self.tfl.lookupTransform(src, tgt, Time())
                self._is_obs_ready[nbhr] = True
                nbhrs.pop()
            except:
                print('unable2tf b/w {} & {}'.format(nbhr, self.ns))
                continue
        obs.update(self.tf_obs)
        print('{}: Time taken for obs: {}'.format(self.ns, time.time() - t))
        if apply_preprocessor and self.preprocessor is not None:
            return self.preprocessor(obs)
        return obs


    @staticmethod
    def laserscan_preprocessor(x):
        return np.tanh(x)

    @staticmethod
    def tf_preprocessor(x):
        trans, rot = x
        return np.array(list(trans[:2]) + list(q2e(rot)[-1:]))
