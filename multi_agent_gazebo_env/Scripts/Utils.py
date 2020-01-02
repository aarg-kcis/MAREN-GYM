import rospy

def get_configuration(env_path):
    try:
        with open(os.path.join(env_path, "config.yaml")) as c_file:
            config = yaml.load(c_file)
        return config
    except Exception as e:
        print ("Problem in loading/parsing config file ...")
        print ("Reading config file at: \n{}".format(env_path+"/config.yaml"))
