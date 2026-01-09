import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser( description="This script demonstrates adding a custom robot to an Isaac Lab environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


OMNIBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"/home/robot/SwitchRobot/summit_xl_omni_four.usd",
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #     rigid_body_enabled=True,
            #     max_linear_velocity=1000.0,
            #     max_angular_velocity=1000.0,
            #     max_depenetration_velocity=100.0,
            #     enable_gyroscopic_forces=True,
            # ),
            #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            #         enabled_self_collisions=False,
            #         solver_position_iteration_count=4,
            #         solver_velocity_iteration_count=0,
            #         sleep_threshold=0.005,
            #         stabilization_threshold=0.001,
            # ),
        ),
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

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = OMNIBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["fl_joint","fr_joint","rl_joint","rr_joint"], scale=5.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    # add_pole_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
    #         "mass_distribution_params": (0.1, 0.5),
    #         "operation": "add",
    #     },
    # )

    # # on reset
    # reset_cart_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.1, 0.1),
    #     },
    # )

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.125 * math.pi, 0.125 * math.pi),
    #         "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
    #     },
    # )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)

            joint_efforts = torch.tensor([[-2.5, -2.5, 2.5,2.5]]) #右前，左前，左后，右后
            print(joint_efforts)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()















# class NewRobotsSceneCfg(InteractiveSceneCfg):
#     """Designs the scene."""
#     ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
#     dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
#     Omnibot = OMNIBOT_CONFIG.replace(prim_path="/summit_xl")

# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     while simulation_app.is_running():
#         # reset
#         if count % 500 == 0:
#             # reset counters
#             count = 0
#             # reset the scene entities to their initial positions offset by the environment origins
#             root_Omnibot_state = scene["Omnibot"].data.default_root_state.clone()
#             print(root_Omnibot_state)
#             root_Omnibot_state[:, :3] += scene.env_origins

#             # copy the default root state to the sim for the jetbot's orientation and velocity
#             scene["Omnibot"].write_root_pose_to_sim(root_Omnibot_state[:, :7])
#             scene["Omnibot"].write_root_velocity_to_sim(root_Omnibot_state[:, 7:])

#             # copy the default joint states to the sim
#             joint_pos, joint_vel = (
#                 scene["Omnibot"].data.default_joint_pos.clone(),
#                 scene["Omnibot"].data.default_joint_vel.clone(),
#             )
#             scene["Omnibot"].write_joint_state_to_sim(joint_pos, joint_vel)

#             # clear internal buffers
#             scene.reset()
#             print("[INFO]: Resetting Jetbot and Dofbot state...")

#         # drive around
#         # if count % 100 < 75:
#         #     # Drive straight by setting equal wheel velocities
#         #     # action = torch.Tensor([[-5, 5, 5, -5, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0]])
#         #     # 0: 左后轮 1: 左前轮 2: 右前轮 3: 右后轮 （0&1：右手定则 方向 ）

#         #     # action = torch.Tensor([[0, -10, 0, 10, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0]])
#         #     action = torch.Tensor([[5, -5, -5, 5, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0]])
#         # else:
#         #     # Turn by applying different velocities
#         #     action = torch.Tensor([[5, -5, -5, 5, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0]])
#         # action = torch.Tensor([[0, -3, 0, 3, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0]])
        
#         # action = torch.Tensor([[0, -3, 0, 3, None, None, None, None,  
#         #                         None, None, None, None, None, None, None, None,
#         #                         None, None, None, None, None, None, None, None,
#         #                         None, None, None, None, None, None, None, None
#         #                         ]])

#         # action = torch.Tensor([[0, 0, 0, 0, 111110, 111110, 11110, 111110,  111110, 11110, 11110, 1110, 11110, 11110, 11110, 11110, 11110, 11110, 11110, 11110, 11110, 11110, 11110, 11110,  11110, 11110, 11110, 11110, 11110, 11110, 11110, 1110]])
#         action = scene["Omnibot"].data.default_joint_vel
#         action[:, 0] = 5
#         action[:, 1] = -5
#         action[:, 2] = -5
#         action[:, 3] = 5
#         scene["Omnibot"].set_joint_velocity_target(action)

#         scene.write_data_to_sim()
#         sim.step()
#         sim_time += sim_dt
#         count += 1
#         scene.update(sim_dt)





# def main():
#     """Main function."""
#     # Initialize the simulation context
#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim = sim_utils.SimulationContext(sim_cfg)
#     sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
#     # Design scene
#     scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)
#     # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # Run the simulator
#     run_simulator(sim, scene)


# if __name__ == "__main__":
#     main()
#     simulation_app.close()






















    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.0),  
    #     rot=(1.0, 0.0, 0.0, 0.0),  
    #     joint_pos={
    #         "fl_joint": 0.0,
    #         "fr_joint": 0.0,
    #         "rl_joint": 0.0,
    #         "rr_joint": 0.0,
    #         "RevoluteJoint": 0.0,
    #         "RevoluteJoint0": 0.0,
    #         "RevoluteJoint1": 0.0,
    #         "RevoluteJoint2": 0.0,
    #         "RevoluteJoint3": 0.0,
    #         "RevoluteJoint4": 0.0,
    #         "RevoluteJoint5": 0.0,
    #         "RevoluteJoint_0": 0.0,
    #         "RevoluteJoint0_0": 0.0,
    #         "RevoluteJoint1_0": 0.0,
    #         "RevoluteJoint2_0": 0.0,
    #         "RevoluteJoint3_0": 0.0,
    #         "RevoluteJoint4_0": 0.0,
    #         "RevoluteJoint5_0": 0.0,
    #         "RevoluteJoint_1": 0.0,
    #         "RevoluteJoint0_1": 0.0,
    #         "RevoluteJoint1_1": 0.0,
    #         "RevoluteJoint2_1": 0.0,
    #         "RevoluteJoint3_1": 0.0,
    #         "RevoluteJoint4_1": 0.0,
    #         "RevoluteJoint5_1": 0.0,
    #         "RevoluteJoint_2": 0.0,
    #         "RevoluteJoint0_2": 0.0,
    #         "RevoluteJoint1_2": 0.0,
    #         "RevoluteJoint2_2": 0.0,
    #         "RevoluteJoint3_2": 0.0,
    #         "RevoluteJoint4_2": 0.0,
    #         "RevoluteJoint5_2": 0.0,
    #     },
    #     joint_vel={
    #         "fl_joint": 0.0,
    #         "fr_joint": 0.0,
    #         "rl_joint": 0.0,
    #         "rr_joint": 0.0,
    #         "RevoluteJoint": 0.0,
    #         "RevoluteJoint0": 0.0,
    #         "RevoluteJoint1": 0.0,
    #         "RevoluteJoint2": 0.0,
    #         "RevoluteJoint3": 0.0,
    #         "RevoluteJoint4": 0.0,
    #         "RevoluteJoint5": 0.0,
    #         "RevoluteJoint_0": 0.0,
    #         "RevoluteJoint0_0": 0.0,
    #         "RevoluteJoint1_0": 0.0,
    #         "RevoluteJoint2_0": 0.0,
    #         "RevoluteJoint3_0": 0.0,
    #         "RevoluteJoint4_0": 0.0,
    #         "RevoluteJoint5_0": 0.0,
    #         "RevoluteJoint_1": 0.0,
    #         "RevoluteJoint0_1": 0.0,
    #         "RevoluteJoint1_1": 0.0,
    #         "RevoluteJoint2_1": 0.0,
    #         "RevoluteJoint3_1": 0.0,
    #         "RevoluteJoint4_1": 0.0,
    #         "RevoluteJoint5_1": 0.0,
    #         "RevoluteJoint_2": 0.0,
    #         "RevoluteJoint0_2": 0.0,
    #         "RevoluteJoint1_2": 0.0,
    #         "RevoluteJoint2_2": 0.0,
    #         "RevoluteJoint3_2": 0.0,
    #         "RevoluteJoint4_2": 0.0,
    #         "RevoluteJoint5_2": 0.0,
    #     }
    # ),