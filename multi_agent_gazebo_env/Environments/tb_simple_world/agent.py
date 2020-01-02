
class Agent(object):
    def __init__(self, model, id, parent):
        # TODO: input topic names from config
        self.model = model
        self._id = id
        self.parent = parent
        self.model_name = "{}_{}".format(self.model, self._id)
        self.ns = "{}".format(self.model_name)
        self.vel_pub = Publisher(self.ns + "/cmd_vel", Twist, queue_size=5)
        self.state_pub = Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=2
        )
        self.tf_listener = TransformListener(False, Duration(5.0))
        self.t_scale_f  = self.parent.t_scale_f
        self.r_scale_f  = self.parent.r_scale_f
        self.act_low    = np.array([-0.1, -np.pi/4])
        self.act_high   = np.array([0.25,  np.pi/4])

    def reset(self, pose):
        # TODO handle reset of ros time?
        reset_state = ModelState()
        reset_state.model_name = self.model_name
        reset_state.pose.position.x = pose[0]
        reset_state.pose.position.y = pose[1]
        self.state_pub.publish(reset_state)
        self.previous_act = [0., 0.]
        self.previous_obs = None

    def get_relative_state(self):
        # the source and target frames have been changed here b/c
        # we dont want to find the position of nhbr wrt to self's frame
        # Maybe implement twist transform ... 
        state = {}
        for nhbr in self.neighbours:
            source_frame = "{}_{}/base_link".format(self.model, self._id)
            target_frame = "{}_{}/base_link".format(self.model, nhbr)
            Ptransform = self.tf_listener.lookupTransform(source_frame, target_frame, Time())
            state[nhbr] = Ptransform
        return state

    def step(self, action):
        # Scale down actions !!!
        # -1 is subtracted as it is the lower limit of the current range
        action = (action -- 1)/2. *(self.act_high-self.act_low) + self.act_low

        try:
          self.previous_obs
          self.previous_act = action
        except:
          print("Can't call step before calling reset() ...")
        vel_cmd = Twist()
        vel_cmd.linear.x, vel_cmd.angular.z = action
        self.vel_pub.publish(vel_cmd)
        return self.get_obs()

    def get_obs(self):
        self.unpause_sim()
        obs = []
        states = self.get_relative_state()
        self.pause_sim()
        for neighbour, state in states.items():
          translation = np.array(state[0])[:-1]
          rotation    = np.array(q2e(state[1])[-1]) # extract only YAW
          # print (self.goal)
          obs.append(np.hstack([translation, rotation, self.goal[neighbour], self.previous_act]))
        obs = self.scale_obs(np.hstack(obs))
        self.previous_obs = deepcopy(obs)
        obs = { "observation": self.previous_obs,
                "achieved_goal": self.previous_obs[self.parent.goal_indices],
                "desired_goal": np.hstack(self.goal.values())}
        return obs
