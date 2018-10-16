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

np.set_printoptions(precision=3, linewidth=150, suppress=True)
class KobukiFormationEnv(MultiAgentGazeboEnv):

  def __init__(self):
    self.env_name = "KobukiFormationEnv"
    self.env_path = os.path.dirname(os.path.abspath(__file__))
    super(KobukiFormationEnv, self).__init__()
    self.processes = []
    self.agent_envs = [KobukiAgentEnv(self.config["agent_ns"], i, self) \
                        for i in range(self.config["num_agents"])]
    self.init_agent_neighbours()
    self.action_dim = 2
    self.goal_dim   = 2
    # DEFINE ENV RELATED VARIABLES LIKE ACTION/ OBS SPACE

  def reset(self):
    poses = self.sample_poses()
    goals = self.sample_goals()
    for agent_env in self.agent_envs:
      agent_id = agent_env._id
      agent_env.reset(poses[agent_id], goals[agent_id])

  def sample_goals(self):
    return {0: {1: [ 0.0 ,    0.75],
                2: [ 0.6495,  0.375]},
            1: {0: [ 0.0,    -0.75],
                2: [ 0.6495, -0.375]},
            2: {0: [-0.6495, -0.375],
                1: [-0.6495,  0.375]}}

  def close(self):
    self.agent_envs[0].unpause_sim()
    return [i.kill() for i in self.processes] 

  def sample_poses(self):
    poses = {0: [0., 0.]}
    while len(poses) < 3:
      rc = np.random.random(2)
      if np.min([np.linalg.norm(i-rc) for i in poses.values()]) > 0.5:
        poses[len(poses)] = rc
    return poses

  # def make_env(self):
  #   pass

  def get_reward(self, c_state, action, a_state):
    pass

  def init_agent_neighbours(self):
    for agent in self.agent_envs:
      agent.neighbours = [i._id for i in self.agent_envs if \
                          i._id != agent._id]

  def subtract_goals(self, d_goal, a_goal):
    diff = d_goal - a_goal
    return np.reshape(diff, -1)

class KobukiAgentEnv(object):
  
  def __init__(self, model, id, parent):
    self.model = model
    self._id = id
    self.parent = parent
    self.model_name = "{}_{}".format(self.model, self._id)
    self.ns = "/{}".format(self.model_name)
    self.vel_pub = Publisher(self.ns + "/mobile_base/commands/velocity", Twist, queue_size=5)
    self.state_pub = Publisher("/gazebo/set_model_state", ModelState, queue_size=2)
    self.tf_listener = TransformListener(False, Duration(5.0))
    self.init_service_proxies()
    self.start_tf_broadcaster()

    ## previous velocities

  def reset(self, pose, goals):
    # TODO handle reset of ros time?
    self.unpause_sim()
    reset_state = ModelState()
    reset_state.model_name = self.model_name
    reset_state.pose.position.x = pose[0]
    reset_state.pose.position.y = pose[1]
    self.state_pub.publish(reset_state)
    self.pause_sim()
    self.goal = goals
    for k, val in self.goal.items():
      self.goal[k] = [i/5. for i in val]
    self.previous_act = [0., 0.]
    self.previous_obs = None

  def get_relative_state(self):
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

  def scale_obs(self):
    pass

  def step(self, action):
    try:
      self.previous_obs
    except:
      print("Can't call step before calling reset() ...")
    self.unpause_sim()
    vel_cmd = Twist()
    vel_cmd.linear.x, vel_cmd.angular.z = action
    self.vel_pub.publish(vel_cmd)
    self.pause_sim()
    return get_obs()

  def get_obs(self):
    obs = []
    states = self.get_relative_state()
    for neighbour, state in states.items():
      translation = np.array(np.clip(state[0], a_min=-5., a_max=5.)[:-1])/5.
      rotation    = np.array(q2e(state[1])[-1])/ np.pi # extract only YAW
      obs.append(np.hstack([translation, rotation, self.goal[neighbour], self.previous_act]))
    obs = np.hstack(obs)
    self.previous_obs = deepcopy(obs)
    obs = { "observation": self.previous_obs,
            "achieved_goal": self.get_achieved_goal(),
            "desired_goal": np.hstack(self.goal.values())}
    return obs

  def get_achieved_goal(self):
    return np.hstack([self.previous_obs[:7][:2], self.previous_obs[7:][:2]])

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
    