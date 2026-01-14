from OmniRobot.robots.omnirobot import OMNIROBOT_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class OmniRobotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 4
    # observation_space = 9
    observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = OMNIROBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    
    dof_names = [
        "fl_joint","fr_joint","rl_joint","rr_joint",
        "RevoluteJoint", "RevoluteJoint0", "RevoluteJoint1", "RevoluteJoint2", "RevoluteJoint3", "RevoluteJoint4", "RevoluteJoint5",
        "RevoluteJoint_0", "RevoluteJoint0_0", "RevoluteJoint1_0", "RevoluteJoint2_0", "RevoluteJoint3_0", "RevoluteJoint4_0", "RevoluteJoint5_0",
        "RevoluteJoint_1", "RevoluteJoint0_1", "RevoluteJoint1_1", "RevoluteJoint2_1", "RevoluteJoint3_1", "RevoluteJoint4_1", "RevoluteJoint5_1",
        "RevoluteJoint_2", "RevoluteJoint0_2", "RevoluteJoint1_2", "RevoluteJoint2_2", "RevoluteJoint3_2", "RevoluteJoint4_2", "RevoluteJoint5_2"
    ]

