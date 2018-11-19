import os
import sys
import numpy as np
import subprocess as sp
from copy import deepcopy

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, PointStamped
from tf import TransformListener
from tf.transformations import euler_from_quaternion as q2e
from rospy import Publisher, Subscriber, ServiceProxy, Time, Duration

import gym
from gym.spaces import Box


class KobukiEnv(object):
    """ Kobuki Percieved Environment:
    Its the environment as percieved by this kobuki agent.
    Observations and actions are relative to this kobuki agent.
    NOTE:   In multiagent setting this environment needs to be made
            using gym.envs.registration.EnvSpec class with diffrent
            agent ids mentioned in config
    
    config is a dictionary containing the following attributes:
    _id (int):  The id for this relative env. This must be unique in a
                multi-agent setting

    """
    def __init__(self, config=None):
        self._id = 0
        self.model = "kobuki"
        self.name = "{}_{}".format(self.model, self._id)
        self.ns = "/{}".format(name)
        self.act_lo = [-0.1, -np.pi/4]
        self.act_hi = [0.33, np.pi/4]
        self.neighbours = None

        # Override params with those mentioned in config
        self.__dict__.update(self.config)
        self.act_lo = np.array(self.act_lo)
        self.act_hi = np.array(self.act_hi)

        self.make_publishers()
        self.make_subscribers()
        self.start_tf_broadcaster()
        self.tf_listener  = TransformListener(False, Duration(5.0))

    def step(self, action):
        if self.prv_obs is None:
            print("Call reset before calling step ...")
            return
        vel_cmd = Twist()
        vel_cmd.linear.x, vel_cmd.angular.z = action
        self.v_pub.publish(vel_cmd)
        self.prv_act = action
        obs = self.get_obs()
        obs.update(self.compute_goal())
        return obs

    def reset(self, goal_point):
        self.prv_obs = None
        self.prv_act = [0. ,0.]
        self.goal_point = goal_point
        obs = self.get_obs()
        obs.update(self.compute_goal())
        return obs

    def compute_goal(self)
        goal = PointStamped()
        goal.point.x, goal.point.y = goal_point
        goal = self.tf_listener.transformPoint("world", goal_point)
        return {"desired_goal": [goal.point.x, goal.point.y]}

    def get_state(self, target, transform_only=False):
        topic = "{}/base_link"
        source = topic.format(self.name)
        target = topic.format(target)
        self.tf_listener.waitForTransform(source, target, Time(),
                                          Duration(100))
        transform = self.tf_listener.lookupTransform(source, target, Time())
        if transform_only:
            return transform
        yaw = np.array(q2e(transform[1]))[-1]
        rotation_v = [np.cos(yaw), np.sin(yaw)]
        translation = np.array(transform[0])[:-1]
        return np.hstack([rotation_v, translation])

    def get_obs(self):
        # OBS
        # self.n1_pose | 4 | sin, cos, x, y # neighbour1
        # self.n2_pose | 4 | sin, cos, x, y # neighbour2
        # self.prv_act | 2 | v, w
        obs = [self.get_state(nbr) for nbr in self.neighbours]
        obs.append(self.prv_act)
        return obs

    def make_publishers(self):
        v_topic = "{}/mobile_base/commands/velocity"
        self.v_pub = Publisher(v_topic.format(self.ns),
                               Twist, queue_size=5)

    def make_subscribers(self):
        pose_topic = "/{}/ground_truth_pose"
        Subscriber(pose_topic.format(self.name),
                   Odometry, self.update, "self")
        for i in self.neighbours:
            Subscriber(pose_topic.format(i), Odometry, self.update, i)

    def start_tf_broadcaster(self):
        print ("Starting TF Broadcaster for {} ...".format(self.name))
        p = sp.Popen([sys.executable, 
                      "{}/KobukiTfBroadcaster.py".format(os.getcwd()), 
                      self.name])
        return p
