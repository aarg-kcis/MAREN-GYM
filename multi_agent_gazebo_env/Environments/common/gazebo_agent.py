import pdb
import rospy
import rostopic
import threading
import numpy as np
from gym.spaces import Dict, Box
from collections import defaultdict, OrderedDict

from utils import *

class GazeboAgent:
    def __init__(self, agent_type, _id, config):
        self.type = agent_type
        self.id = _id
        self.ns = '{}_{}'.format(agent_type, _id)
        self.config = config
        self._actions = OrderedDict()
        self._publishers = OrderedDict()
        self._observations = OrderedDict()
        self.reset()
        self.preprocessor = None
        self._configure_actions()
        self._configure_observations()
        # check_for_ROS()

        print('='*100)
        print(self)
        print(rospy.get_name())
        print(rospy.get_namespace())
        print(rospy.get_node_uri())
        print('='*100)

    def reset(self):
        print(self, 'RESET')
        self._is_obs_ready = dict.fromkeys(self._observations, False)

    def step(self, cmd: dict):
        print(self, 'STEP')
        cmd = self.map_actions(cmd)
        for action_name, val in self._actions.items():
            s, msg = val['set'], val['msg']
            for v in s:
                set_data(msg, v.split('.'), cmd[action_name][v])
            self._publishers[action_name].publish(msg)

    def is_obs_ready(self):
        return all([v for k, v in self._is_obs_ready.items()
                    if k in self._observations.keys()])

    def get_obs(self, apply_preprocessor=True):
        print(self, 'GET_OBS')
        obs = {}
        while not self.is_obs_ready():
            pass
        print(f'# {self.ns}', self._is_obs_ready)
        self._is_obs_ready = dict.fromkeys(self._observations, False)
        for obs_name, val in self._observations.items():
            obs[obs_name] = np.array(
                [get_data(val['msg'], s.split('.')) for s in val['store']]
            ).squeeze()
        if apply_preprocessor and self.preprocessor is not None:
            return self.preprocessor(obs)
        return obs

    def map_actions(self, x):
        print(self, 'MAP_ACTIONS')
        assert self.action_space.contains(x)
        ctrx = 0
        ctrl = dict.fromkeys(self._actions, None)
        for action_name, val in self._actions.items():
            ctrl[action_name] = val['map'](x[ctrx: ctrx + val['spc'].shape[0]])
        return ctrl

    def get_random_action(self):
        random_action = self.action_space.sample()
        return self.map_actions(random_action)

    def _configure_actions(self):
        for action_name, val in self.config['action_spaces'].items():
            try:
                self._create_publisher(val, action_name) 
            except Exception as e:
                rospy.logerr(e)
                rospy.logerr(
                    'Error creating publisher for {}:'.format(action_name))
                rospy.logerr(val)
        self.action_space_maren = Dict({
            k: v['spc'] for k, v in self._actions.items()
        })
        low = np.concatenate([v['spc'].low for v in self._actions.values()])
        high = np.concatenate([v['spc'].high for v in self._actions.values()])
        self.action_space = Box(low=low, high=high)

    def _configure_observations(self):
        for obs_name, val in self.config['observation_spaces'].items():
            try:
                self._create_subscriber(val, obs_name)
            except Exception as e:
                rospy.logerr(e)
                rospy.logerr(
                    'Error creating subscriber for {}:'.format(obs_name))
                rospy.logerr(val)
        self.observation_space = Dict({
            k: v['spc'] for k, v in self._observations.items()
        })

    def _prepare_topic_name(self, x):
        x = x.replace('$NS', self.ns)
        if x[0] != '/':
            rospy.logwarn('Adding preceeding / to topicname [{}]'.format(x))
            x = '/{}'.format(x)
        return x

    def _create_publisher(self, val, action_name):
        queue_size = val.get('queue_size', 10)
        topic_name = self._prepare_topic_name(val['topic'])
        topic_type = rostopic.get_topic_class(topic_name)[0]
        var_to_set = val.get('set', None)
        if var_to_set is None:
            raise ValueError('"set" should be provided!!')
            var_to_set = get_all_rosmsg_vars(topic_type)
        space, map_fn = get_gym_space_with_map(val, action_name)
        self._actions[action_name] = {
            'set': var_to_set,
            'msg': topic_type(),
            'spc': space,
            'map': map_fn
        }
        p = rospy.Publisher(topic_name, topic_type, queue_size=queue_size)
        self._publishers[action_name] = p

    def _create_subscriber(self, val, obs_name):
        topic_name = self._prepare_topic_name(val['topic'])
        topic_type = rostopic.get_topic_class(topic_name)[0]
        space, map_fn = get_gym_space_with_map(val, obs_name)
        self._observations[obs_name] = {
            'store': val['store'],
            'msg': None,
            'map': map_fn,
            'spc': space
        }
        self._is_obs_ready[obs_name] = False
        def callback(msg):
            self._observations[obs_name]['msg'] = msg
            self._is_obs_ready[obs_name] = True
        ss = rospy.Subscriber(topic_name, topic_type, callback)
        print('Subscriber for {} made.'.format(topic_name))

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    from utils import get_configuration
    import pdb
    rospy.init_node('GazeboAgentNode')
    rosrate = rospy.Rate(5) # 10hz
    config = get_configuration('../tb_simple_world')
    agent = GazeboAgent('waffle_1', config['agents']['waffle'])
    topic_name = '/waffle_1/cmd_vel'
    topic_type = rostopic.get_topic_class(topic_name)[0]
    s = topic_type()
    s.linear.x = 10
    with open('a.txt', 'w+') as f:
        s.serialize_numpy(f, np)
    pdb.set_trace()
    # while not rospy.is_shutdown():
    #     print(agent._get_obs())
    #     rosrate.sleep()
