import os
import sys
import numpy as np
import subprocess as sp
from copy import deepcopy

import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from tf import TransformListener
from tf.transformations import euler_from_quaternion as q2e
from rospy import Publisher, Subscriber, ServiceProxy, Time, Duration

from MultiAgentGazeboEnv import MultiAgentGazeboEnv
from KobukiEnvironment import KobukiEnv
from gym.spaces import Box


np.set_printoptions(precision=5, linewidth=250, suppress=True)


class KobukiFormationEnv(MultiAgentGazeboEnv):
    """ Multi Agent Go2Goal Environment
    Simple environment where multiple agents are required to reach a
    goal point whilst being in a formation.

    config includes:

    """
    def __init__(self, config=None):
        self.env_name = "KobukiFormation-v0"
        self.env_path = os.path.dirname(os.path.abspath(__file__))
        super(KobukiFormationEnv, self).__init__()
        self.processes = []
        self.max_episode_len = 100
        self.num_agents = 3
        self.agent_ns = "kobuki"
        if config is not None:
            self.__dict__.update(config)
        self.state_pub = Publisher("/gazebo/set_model_state",
                                   ModelState, queue_size=2)
        self.agent_envs = [KobukiEnv(config={"_id": i})\
                           for i in range(self.num_agents)]
        self.init_agent_neighbours()
        self.init_service_proxies()


    def init_agent_neighbours(self):
        for agent in self.agent_envs:
            agent.neighbours = [i.name for i in self.agent_envs\
                                if i._id != agent._id]

    def init_service_proxies(self):
        self.unpause = ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = ServiceProxy("/gazebo/reset_simulation", Empty)

    def pause_sim(self, t=None):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

    def unpause_sim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

    def seed(self, seed):
        np.random.seed(seed)
        print("NUMPY RANDOM SEED SET TO {}".format(seed))

    def sample_poses(self, t_max, t_min):
        poses = {0: [0., 0.]}
        while len(poses) < 3:
            rc = np.random.random(2)*(t_max-t_min) + t_min
            if np.min([np.linalg.norm(i-rc) for i in poses.values()]) > 0.8:
                poses[len(poses)] = rc
                continue
            rc = rc[::-1]
            if np.min([np.linalg.norm(i-rc) for i in poses.values()]) > 0.8:
                poses[len(poses)] = rc
        return poses

    def sample_goal(self):
        poses = self.sample_poses(1.5, 0)
        formation_goal = [np.linalg.norm(poses[i]-poses[j])
                          for i in poses.keys()
                          for j in poses.keys() if i < j]
        destination = np.random.random(2)*8 - 4
        return sorted(formation_goal), destination

    def reset(self):
        poses = self.sample_poses(2, -2)
        print("POSES\n",poses)
        self.goal = self.sample_goal()
        obs = {}
        self.unpause_sim()
        for agent_env in self.agent_envs:
            reset_state = ModelState()
            reset_state.model_name = agent_env.name
            reset_state.pose.position.x = poses[agent_env._id][0]
            reset_state.pose.position.y = poses[agent_env._id][1]
            self.state_pub.publish(reset_state)
        
        for agent_env in self.agent_envs:
            obs[agent_env._id] = agent_env.reset(self.goal[1])

        sides_ag = sorted([np.linalg.norm(poses[i]-poses[j])
                           for i in poses.keys()
                           for j in poses.keys() if i < j])


        for _id, ob in obs.items():
            ob["achieved_goal"] = np.hstack([[0., 0.], sides_ag])
            ob["desired_goal"] = np.hstack([ob["desired_goal"], self.goal[0]])

        self.pause_sim()
        return obs
        # return {k: np.hstack([v[k] for v in obs]) for k in self.obs_keys}
        # Or maybe return agent eise observations
        # return obs

    def step(self, actions):
        actions = {agent._id: actions[agent._id]
                   if agent._id in actions.keys() else [0, 0]
                   for agent in self.agent_envs}
        obs = []
        self.unpause_sim()
        for agent_env in self.agent_envs:
            obs.append(agent_env.step(actions[agent_env._id]))
        self.pause_sim()
        return obs

    def close(self):
        self.agent_envs[0].unpause_sim()
        return [i.kill() for i in self.processes]

    def get_reward(self, curr_s, action, acq_s):
        pass