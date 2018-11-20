import os
import sys
import numpy as np
import subprocess as sp
from copy import deepcopy

import rospy
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
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
        self.act_lo = [-0.1, -np.pi/4]
        self.act_hi = [0.33, np.pi/4]
        self.neighbours = None

        # Override params with those mentioned in config
        if config is not None:
            self.__dict__.update(config)
        self.act_lo = np.array(self.act_lo)
        self.act_hi = np.array(self.act_hi)
        self.name = "{}_{}".format(self.model, self._id)
        self.ns = "/{}".format(self.name)
        self.prv_act = None

        self.make_publishers()
        # self.make_subscribers()
        self.start_tf_broadcaster()
        self.tf_listener  = TransformListener(False, Duration(5.0))

    def step(self, action):
        if self.prv_act is None:
            print("Call reset before calling step ...")
            return
        self.prv_transform = self.get_state("world", True)
        vel_cmd = Twist()
        vel_cmd.linear.x, vel_cmd.angular.z = action
        self.v_pub.publish(vel_cmd)
        self.prv_act = action
        # transformation from this frame to world frame
        print(self.prv_transform)
        obs = {"observation": self.get_obs()}
        obs.update(self.compute_goal())
        return obs

    def reset(self, goal_point):
        self.prv_obs = None
        self.prv_act = [0. ,0.]
        self.goal_point = goal_point
        obs = {"observation": self.get_obs()}
        obs.update(self.compute_goal())
        return obs

    def compute_goal(self):
        goal = PointStamped()
        goal.header.frame_id = "world"
        goal.header.stamp = rospy.Time(0)
        goal.point.x, goal.point.y = self.goal_point
        goal = self.tf_listener.transformPoint(self.name+"/base_link", goal)
        return {"desired_goal": [goal.point.x, goal.point.y],
                "achieved_goal": [0., 0.]}

    def get_state(self, target, transform_only=False):
        topic = "{}/base_link"
        source = topic.format(self.name)
        if not transform_only:
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
        return np.hstack(obs)

    def make_publishers(self):
        v_topic = "{}/mobile_base/commands/velocity"
        self.v_pub = Publisher(v_topic.format(self.ns),
                               Twist, queue_size=1)

    def make_subscribers(self):
        pose_topic = "/{}/ground_truth_pose"
        Subscriber(pose_topic.format(self.name),
                   Odometry, self.update, "self")
        for i in self.neighbours:
            Subscriber(pose_topic.format(i), Odometry, self.update, i)

    def start_tf_broadcaster(self):
        print ("Starting TF Broadcaster for {} ...".format(self.name))
        p = sp.Popen([sys.executable, 
                     "{}/KobukiTfBroadcaster.py".format(os.path.dirname
                     (os.path.realpath(__file__))), 
                      self.name])
        return p
