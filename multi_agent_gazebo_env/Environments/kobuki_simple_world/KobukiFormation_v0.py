import os
import sys
import numpy as np
import subprocess as sp
from copy import deepcopy

from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState

import rospy
from tf import TransformListener
from tf.transformations import euler_from_quaternion as q2e
from rospy import Publisher, Subscriber, ServiceProxy, Time, Duration

from ..MultiAgentGazeboEnv import MultiAgentGazeboEnv

class KobukiFormationEnv(MultiAgentGazeboEnv):

  def __init__(self):
    self.env_name = "KobukiFormationEnv"
    self.env_path = os.path.dirname(os.path.abspath(__file__))
    super(KobukiFormationEnv, self).__init__()
    self.processes = []
    self.agents_envs = [KobukiAgentEnv(self.config["agent_ns"], i, self) \
                        for i in range(self.config["num_agents"])]
    self.init_agent_neighbours()
    # DEFINE ENV RELATED VARIABLES LIKE ACTION/ OBS SPACE

  def reset(self):
    if np.array([i.locked for i in self.agents_envs]).all():
      print "hooray!"
      for i in self.agents_envs:
        i.locked = False

  def sample_goal(self):
    pass

  def close(self):
    pass

  def gen_random_poses(self):
    pass

  def make_env(self):
    pass

  def get_reward(self, c_state, action, a_state):
    pass

  def init_agent_neighbours(self):
    for agent in self.agents_envs:
      agent.neighbours = [i.model_name for i in self.agents_envs if \
                          i.model_name != agent.model_name]
        


class KobukiAgentEnv(object):
  
  def __init__(self, model, id, parent):
    self.model = model
    self._id = id
    self.parent = parent
    self.model_name = "{}_{}".format(self.model, self._id)
    self.ns = "/{}".format(self.model_name)
    self.vel_pub = Publisher(self.ns + "/mobile_base/commands/velocity", Twist, queue_size=5)
    self.state_pub = Publisher("/gazebo/set_model_state", ModelState, queue_size=2)
    self.tf_listener = TransformListener(True, Duration(5.0))
    self.previous_obs = None
    self.locked = None
    self.init_service_proxies()
    self.start_tf_broadcaster()

    ## previous velocities

  def reset(self, pose):
    # TODO handle reset of ros time?
    self.unpause_sim()
    self.locked = True
    self.parent.reset()
    # reset_state = ModelState()
    # reset_state.model_name = self.model_name
    # reset_state.pose.position.x = pose["x"]
    # reset_state.pose.position.y = pose["y"]
    # self.state_pub.publish(reset_state)
    # self.pause_sim()

  def get_relative_state(self, nhbrs):
    # the source and target frames have been changed here b/c
    # we dont want to find the position of nhbr wrt to 'this' (self's) frame
    # Maybe implement twist transform ... 
    obs = {}
    for nhbr in nhbrs:
      source_frame = "{}/base_link".format(self.model_name)
      target_frame = "{}/base_link".format(nhbr)
      Ptransform = self.tf_listener.lookupTransform(source_frame, target_frame, Time())
      obs[nhbr] = Ptransform
    return obs

  def step(self, action):
    while self.locked: pass
    if self.previous_obs != None:
      print("Can't call step before calling reset() ...")
    self.unpause_sim()
    vel_cmd = Twist()
    vel_cmd.linear.x, vel_cmd.angular.z = action
    self.vel_pub.publish(vel_cmd)
    self.pause_sim()
    return get_obs()

  def get_obs(self):
    obs = []
    states = self.get_relative_state(self.neighbours)
    for neighbour, state in states.items():
      translation = np.clip(state[0], a_min=-5., a_max=5.)
      rotation    = q2e(state[1])[-1]/ np.pi # extract only YAW

    self.previous_obs = obs.deepcopy()

  def init_service_proxies(self):
    self.unpause = ServiceProxy("/gazebo/unpause_physics", Empty)
    self.pause = ServiceProxy("/gazebo/pause_physics", Empty)
    self.reset_proxy = ServiceProxy("/gazebo/reset_simulation", Empty)

  def pause_sim(self):
    rospy.wait_for_service('/gazebo/pause_physics')
    try:
      self.pause()
    except (rospy.ServiceException) as e:
      print ("/gazebo/pause_physics service call failed")

  def unpause_sim(self):
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
      self.unpause()
    except (rospy.ServiceException) as e:
      print ("/gazebo/unpause_physics service call failed")

  def start_tf_broadcaster(self):
    print ("Starting TF Broadcaster for {} ...".format(self.model_name))
    p = sp.Popen([sys.executable, "{}/KobukiTfBroadcaster.py".format(self.parent.env_path), self.model_name])
    self.parent.processes.append(p)
    