import os
import sys
import gym
import yaml
import time
import rospy
import numpy as np 
import subprocess as sp

sys.path.append('/home/abhay/MAREN-GYM/multi_agent_gazebo_env/Scripts')
import Utils as U

from rosgraph_msgs.msg import Clock

class MultiAgentGazeboEnv(gym.Env):
  def __init__(self):
    print ("ENV PATH: {}".format(self.env_path)) 
    self.config = self.configure()
    try:
      self.ros_master_uri = os.environ['ROS_MASTER_URI']
      rospy.init_node(self.env_name)
      rospy.loginfo("Waiting for clock from ROSMASTER@[{}]".format(self.ros_master_uri))
      rospy.wait_for_message('/clock', Clock, timeout=10)
      rospy.loginfo("Success ...")
    except Exception as e:
      rospy.logerr("Couldn't receive clock messages from ROSMASTER")
      rospy.logerr("Exception: \n{}".format(e))
      rospy.logerr("Can't proceed ...")
      sys.exit(-1)

  def configure(self):
    try:
      with open(os.path.join(self.env_path, "config.yaml")) as c_file:
        config = yaml.load(c_file)
        return config
    except Exception as e:
      print ("Problem in loading/parsing config file ...")
      print ("Looking for file at: \n{}".format(self.env_path+"/config.yaml"))
      print (e)
      raise ValueError

  def step(self):
    raise NotImplementedError

  def reset(self):
    raise NotImplementedError
