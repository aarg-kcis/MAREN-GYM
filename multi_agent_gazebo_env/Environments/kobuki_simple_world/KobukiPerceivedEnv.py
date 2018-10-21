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

from gym.spaces import Box

class KobukiPerceivedEnv(object):
  v_topic = "{}/mobile_base/commands/velocity"

  def __init__(self, model, id, parent):
    self.model        = model
    self._id          = id
    self.parent       = parent
    self.model_name   = "{}_{}".format(self.model, self._id)
    self.ns           = "/{}".format(self.model_name)
    self.vel_pub      = Publisher(v_topic.format(self.ns) , Twist, queue_size=5)
    self.state_pub    = self.parent.state_pub
    self.tf_listener  = TransformListener(False, Duration(5.0))
    self.act_low      = np.array([-0.1, -np.pi/4])
    self.act_high     = np.array([0.25,  np.pi/4])
    self.goal_idxs    = self.parent.goal_indices
    self.pause_sim    = self.parent.pause_sim
    self.unpause_sim  = self.parent.unpause_sim
    self.start_tf_broadcaster()

  def reset(self, pose, goals):
    # TODO handle reset of ros time?
    self.unpause_sim()
    reset_state = ModelState()
    reset_state.model_name = self.model_name
    reset_state.pose.position.x = pose[0]
    reset_state.pose.position.y = pose[1]
    self.state_pub.publish(reset_state)
    self.pause_sim()
    self.goal = np.hstack([goals[i] for i in self.neighbours])
    self.previous_act = [0., 0.]
    self.previous_obs = None
    print ("RESETTING AGENT {}".format(self._id))
    print ("GOAL PROVIDED: {}".format(self.goal))

  def get_transforms(self):
    # the source and target frames have been changed here b/c
    # we dont want to find the position of nhbr wrt to 'this' (self's) frame
    # Maybe implement twist transform ... 
    state = {}
    for nhbr in self.neighbours:
      source_frame = "{}_{}/base_link".format(self.model, self._id)
      target_frame = "{}_{}/base_link".format(self.model, nhbr)
      Ptransform = self.tf_listener.lookupTransform(source_frame, target_frame, Time())
      state[nhbr] = Ptransform
    return state

  def step(self, action):
    try:
      self.previous_obs
    except:
      print("Can't call step before calling reset() ...")
    self.unpause_sim()
    vel_cmd = Twist()
    vel_cmd.linear.x, vel_cmd.angular.z = action
    self.vel_pub.publish(vel_cmd)
    self.previous_act = action
    self.pause_sim()
    return self.get_obs()

  def get_obs(self):
    # observation: [tx, ty , cos(tr), sin(tr), v_(t-1), w_(t-1)] * num_neighbours
    self.unpause_sim()
    obs = []
    states = self.get_transforms()
    self.pause_sim()
    for neighbour, state in states.items():
      translation = np.array(state[0])[:-1]
      rotation    = np.array(q2e(state[1])[-1]) # extract only YAW
      rotation_v  = np.array([np.cos(rotation), np.sin(rotation)]) # extract only YAW
      obs.append(np.hstack([translation, rotation, self.previous_act]))
    self.previous_obs = deepcopy(np.hstack(obs))
    obs = { "observation": self.previous_obs,
            "achieved_goal": self.previous_obs[self.goal_idxs],
            "desired_goal": self.goal}
    return obs

  def start_tf_broadcaster(self):
    print ("Starting TF Broadcaster for {} ...".format(self.model_name))
    p = sp.Popen([sys.executable, "{}/KobukiTfBroadcaster.py".format(self.parent.env_path), self.model_name])
    self.parent.processes.append(p)
    