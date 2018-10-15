from gym.envs.registration import register

register(
    id='KobukiFormation-v0',
    entry_point='multi_agent_gazebo_env.Environments.kobuki_simple_world:KobukiFormationEnv',
    # More arguments here
)