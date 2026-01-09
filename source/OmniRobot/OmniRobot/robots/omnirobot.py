import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

OMNIROBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"/root/IsaacLab/scripts/OmniRobot-RL/summit_xl_omni_four.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[
                "fl_joint","fr_joint","rl_joint","rr_joint"
                ], 
            damping=None, 
            stiffness=None
            ),

        "revolute_group": ImplicitActuatorCfg(
            joint_names_expr=[
                "RevoluteJoint", "RevoluteJoint0", "RevoluteJoint1", "RevoluteJoint2", "RevoluteJoint3", "RevoluteJoint4", "RevoluteJoint5",
                "RevoluteJoint_0", "RevoluteJoint0_0", "RevoluteJoint1_0", "RevoluteJoint2_0", "RevoluteJoint3_0", "RevoluteJoint4_0", "RevoluteJoint5_0",
                "RevoluteJoint_1", "RevoluteJoint0_1", "RevoluteJoint1_1", "RevoluteJoint2_1", "RevoluteJoint3_1", "RevoluteJoint4_1", "RevoluteJoint5_1",
                "RevoluteJoint_2", "RevoluteJoint0_2", "RevoluteJoint1_2", "RevoluteJoint2_2", "RevoluteJoint3_2", "RevoluteJoint4_2", "RevoluteJoint5_2"
                ], 
            damping=0.0, 
            stiffness=0.0
            )
    },
)