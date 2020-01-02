import os
import gym
import yaml
import rospy
import numpy as np


def get_configuration(env_path):
    try:
        with open(os.path.join(env_path, "config.yaml")) as c_file:
            config = yaml.load(c_file, Loader=yaml.SafeLoader)
            return config
    except Exception as e:
        print("Problem in loading/parsing config file ...")
        print("Reading config file at: \n{}".format(env_path+"/config.yaml"))
        print(e)


def get_all_rosmsg_vars(msg_type, x=None):
    """
    Return all the set-able params in a list
    eg: for geometry_msg/Twist should return
    ['linear.x', 'linear.y', 'linear.z', 'angular.x', 'angular.y', 'angular.z']
    
    """
    raise NotImplementedError


def get_bounds(sconf, which, shape):
    if isinstance(sconf[which], dict):
        bounds = np.array(list(sconf[which].values()))
    elif isinstance(sconf[which], list):
        bounds = np.array(sconf[which])
    else:
        bounds = np.full(shape, sconf[which])
    return bounds


def mapping_fn_generator(var_list, space):
    def one2one(x):
        return dict(zip(var_list, x))

    def single(x):
        return dict(zip(var_list, x))

    if len(var_list) == space.shape[0]:
        return one2one
    elif len(var_list) == 1:
        return single
    else:
        raise ValueError('Something is wrong in generating mapping_fn')


def get_gym_space_with_map(val, action_name):
    # assert 'space' in val.keys() and 'type' in val['space'].keys(), \
    #     'Invalid space config for action: [{}]'.format(action_name)
    sconf = val['space']
    space = getattr(gym.spaces, sconf['type'])
    vlist = val.get('set', val.get('store'))
    shape = sconf.get('size', len(vlist))
    lower = get_bounds(sconf, 'low', shape)
    upper = get_bounds(sconf, 'high', shape)
    s = space(low=lower, high=upper, dtype=np.float32)
    return s, mapping_fn_generator(vlist, s)


def check_for_ROS():
    from rosgraph_msgs.msg import Clock
    try:
        rospy.loginfo("Waiting for rosmaster ...")
        rospy.wait_for_message('/clock', Clock, timeout=10)
        rospy.loginfo("Success ...")
    except Exception as e:
        rospy.logerr("Couldn't receive clock messages from rosmaster.")
        rospy.logerr("Make sure you have run MAREN and sourced ros.")
        rospy.logerr("Exception: \n{}".format(e))
        rospy.logerr("Can't proceed ...")


def set_data(x, s, v):
    if len(s) == 1: 
        setattr(x, s[0], v) 
    else: 
        set_data(getattr(x, s[0]), s[1:], v)


def get_data(x, s):
    if len(s) == 1: 
        return getattr(x, s[0]) 
    else: 
        return get_data(getattr(x, s[0]), s[1:])

