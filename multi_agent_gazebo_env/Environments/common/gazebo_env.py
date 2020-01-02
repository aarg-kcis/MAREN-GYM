import os
import sys
import gym
import yaml
import time
import enum
import rospy
import numpy as np 
import subprocess as sp

from gym.spaces import Dict
from utils import check_for_ROS
from gazebo_agent import GazeboAgent


from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from tf.transformations import quaternion_from_euler as e2q
from rospy import Publisher, Subscriber, ServiceProxy, Time, Duration

class GazeboState(enum.Enum):
    PAUSED = 0
    UNPAUSED = 1
    INIT = -1

class GazeboEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        node_name = config.get('node_name', 'GazeboEnvNode')
        rospy.init_node(node_name, disable_signals=True)
        self.gz_state = GazeboState.INIT
        self.config = config
        self.state_pub = Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10
        )
        self.pause = ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause = ServiceProxy("/gazebo/unpause_physics", Empty)
        self.reset_proxy = ServiceProxy("/gazebo/reset_simulation", Empty)
        self.add_agents()
        time.sleep(0.2)
        self.action_space = Dict({
            k: v.action_space for k, v in self.agents.items()
        })
        self.observation_space = Dict({
            k: v.observation_space for k, v, in self.agents.items()
        })
        self.unpause_gazebo()
        # check_for_ROS()
        self.pause_gazebo()
        print('='*100)
        print(self)
        print(rospy.get_name())
        print(rospy.get_namespace())
        print(rospy.get_node_uri())
        print('='*100)

    def add_agents(self):
        self.agents = {}
        for a_type, agent_config in self.config['agents'].items():
            agent_cls = agent_config['agent_class'] or GazeboAgent
            for i in range(1, agent_config['num_agents']+1):
                agent = agent_cls(a_type, i, agent_config)
                self.agents[agent.ns] = agent

    def reset(self, agent_states=None):
        self.pause_gazebo()
        self.change_agent_states(agent_states)
        self.unpause_gazebo()
        obs = self.get_obs_from_agents()
        self.pause_gazebo()
        return obs

    def step(self, cmd):
        # cmd: <dict> {<agent_name>: {<agent_action>: action_val}}
        self.unpause_gazebo()
        for agent_name, agent in self.agents.items():
            agent.step(cmd[agent_name])
        t = time.time()
        obs = self.get_obs_from_agents()
        print('ENVIRONMENT: Time taken for obs: {}'.format(time.time() - t))
        self.pause_gazebo()
        rew = dict.fromkeys(obs, 0)
        fin = {'__all__': False}
        return obs, rew, fin, {}

    def get_obs_from_agents(self):
        obs = {}
        print({k: a.is_obs_ready() for k, a in self.agents.items()})
        for agent_name, agent in self.agents.items():
            print('Waiting for agent {}'.format(agent_name))
            obs[agent_name] = agent.get_obs()
        print({k: a.is_obs_ready() for k, a in self.agents.items()})
        return obs
        
    def change_agent_states(self, agent_states):
        if agent_states is None:
            return
        for agent_name, agent_state in agent_states.items():
            self.state_pub.publish(agent_state)
            self.agents[agent_name].reset()

    def pause_gazebo(self):
        if self.gz_state == GazeboState.PAUSED:
            return
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
            self.gz_state = GazeboState.PAUSED
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
            print(e)

    def unpause_gazebo(self):
        if self.gz_state == GazeboState.UNPAUSED:
            return
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
            self.gz_state = GazeboState.UNPAUSED
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
            print(e)

    def get_random_state(self):
        ags = {}
        for agent_name in self.agents.keys():
            m = ModelState()
            m.model_name = agent_name
            m.pose.position.x = np.random.random()*10 - 5
            m.pose.position.y = np.random.random()*10 - 5
            ags[agent_name] = m
            continue
            x, y, z, w = e2q(0, 0, np.random.uniform(-np.pi, np.pi))
            m.pose.orientation.x = x
            m.pose.orientation.y = y
            m.pose.orientation.z = z
            m.pose.orientation.w = w
        return ags

    def get_random_action(self):
        return env.action_space.sample()

    def attach_preprocessors(self, preprocessors):
        for agent_name, agent in self.agents.items():
            agent.preprocessor = preprocessors[agent.type]


if __name__ == '__main__':
    from utils import *
    import pdb
    # from gazebo_tb_mpc_env import *
    # rospy.init_node('GazeboAgentNode')
    config = get_configuration('../tb_simple_world')
    # rosrate = rospy.Rate(1) # 10hz
    # config['rr'] = rosrate
    from gazebo_tb_agent import GazeboTurtleBotAgent
    config['agents']['waffle']['agent_class'] = GazeboTurtleBotAgent
    print(config)
    env = GazeboEnv(config)
    o = env.reset(env.get_random_state())
    benv = BaseEnv.to_base_env(env)
    benv.poll()
    # t = time.time()
    # for i in range(100):
    #     env.step(env.get_random_action())
    # print(time.time() - t)

    # print([x['gt_pose'][:2] for x in o.values()])
    # print([env.get_random_action()])
    # print([a.get_random_action() for a in env.agents.values()])
    pdb.set_trace()