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

sys.path.append('/home/abhay/MAREN-GYM/multi_agent_gazebo_env/Environments')
from MultiAgentGazeboEnv import MultiAgentGazeboEnv
from KobukiPerceivedEnv import KobukiPerceivedEnv
from gym.spaces import Box

np.set_printoptions(precision=3, linewidth=150, suppress=True)

class KobukiFormationEnv(MultiAgentGazeboEnv):

  def __init__(self):
    self.env_name = "KobukiFormationEnv"
    self.env_path = os.path.dirname(os.path.abspath(__file__))
    super(KobukiFormationEnv, self).__init__()
    self.processes = []
    self.max_episode_steps = 100
    self.goal_indices = [0, 1, 6, 7]
    self.agent_envs = [KobukiPerceivedEnv(self.config["agent_ns"], i, self) \
                        for i in range(self.config["num_agents"])]
    self.init_agent_neighbours()

  def reset(self):
    poses = self.sample_poses()
    goals = self.sample_goals()
    for agent_env in self.agent_envs:
      agent_id = agent_env._id
      agent_env.reset(poses[agent_id], goals[agent_id])

  def sample_goals(self):
    goal = {0: {1: [ 0.0 ,    0.75],
                2: [ 0.6495,  0.375]},
            1: {0: [ 0.0,    -0.75],
                2: [ 0.6495, -0.375]},
            2: {0: [-0.6495, -0.375],
                1: [-0.6495,  0.375]}}
    return goal

  def close(self):
    self.agent_envs[0].unpause_sim()
    return [i.kill() for i in self.processes] 

  def sample_poses(self):
    np.random.seed(206)
    poses = {0: [0., 0.]}
    while len(poses) < 3:
      rc = np.random.random(2)
      if np.min([np.linalg.norm(i-rc) for i in poses.values()]) > 0.5:
        poses[len(poses)] = rc
    return poses

  def get_reward(self, curr_s, action, acq_s):
    pass

  def init_agent_neighbours(self):
    for agent in self.agent_envs:
      agent.neighbours = [i._id for i in self.agent_envs if \
                          i._id != agent._id]

  def init_service_proxies(self):
    self.unpause = ServiceProxy("/gazebo/unpause_physics", Empty)
    self.pause = ServiceProxy("/gazebo/pause_physics", Empty)
    self.reset_proxy = ServiceProxy("/gazebo/reset_simulation", Empty)  

  def pause_sim(self, t=None):
    if t == None:
      return
    rospy.wait_for_service('/gazebo/pause_physics')
    try:
      self.pause()
    except (rospy.ServiceException) as e:
      print ("/gazebo/pause_physics service call failed")

  def unpause_sim(self):
    return
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
      self.unpause()
    except (rospy.ServiceException) as e:
      print ("/gazebo/unpause_physics service call failed")


  # def subtract_goals(self, d_goal, a_goal):
  #   diff = d_goal - a_goal
  #   return np.reshape(diff, -1)

