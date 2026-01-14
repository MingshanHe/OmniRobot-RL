# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .omnirobot_env_cfg import OmniRobotEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class OmniRobotEnv(DirectRLEnv):
    cfg: OmniRobotEnvCfg

    def __init__(self, cfg: OmniRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.dof_idx, _ = self.robot.find_joints(["fl_joint","fr_joint","rl_joint","rr_joint"])

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # add robot initliazed position
        # self.init_root_pos = self.robot.data.root_pos_w
        self.env_origins = self.scene.env_origins.clone()

        self.visualization_markers = define_markers()

        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)
        
        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5

        self.marker_loc = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_loc[:,-1] = 0.5
        self.marker_loc = self.marker_loc + self.env_origins
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

        # last_distance and current_distance
        self.reset_flag = True
        self.last_distance = None
        self.current_distance = None
        self.success_buffer = torch.zeros(self.cfg.scene.num_envs, device=self.device, dtype=torch.bool)
        self.success_bonus_flag = torch.ones(self.cfg.scene.num_envs, device=self.device, dtype=torch.bool)
        self.success_dist = 0.05
        self.success_vel  = 0.1

        self.reach_wp_buffer = torch.zeros(self.cfg.scene.num_envs, device=self.device, dtype=torch.bool)
        self.reach_wp_bonus_flag = torch.ones(self.cfg.scene.num_envs, device=self.device, dtype=torch.bool)
        self.reach_wp_dist = 0.1

        self.task_failed = torch.zeros((self.cfg.scene.num_envs,), device=self.device, dtype=torch.bool)
        self.success = torch.zeros((self.cfg.scene.num_envs,), device=self.device, dtype=torch.bool)
        self.prev_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        

    def _visualize_markers(self):
        self.forward_marker_locations = self.robot.data.root_pos_w
        self.command_marker_locations = self.marker_loc
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()
        # print(self.env_origins)
        loc_forward = self.forward_marker_locations
        loc_command = self.command_marker_locations
        loc = torch.vstack((loc_forward, loc_command))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()# + torch.ones_like(actions)
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w 
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        # obs = torch.hstack((self.velocity, self.commands))

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        # obs = torch.hstack((dot, cross, forward_speed))

        ang_vel_b = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)
        gyro_z = ang_vel_b[:, 2:3]

        world_accel = (self.robot.data.root_lin_vel_w - self.prev_lin_vel) / self.cfg.sim.dt
        proper_accel_b = math_utils.quat_apply_inverse(self.robot.data.root_quat_w,
                                                  world_accel - torch.tensor([0.0, 0.0, -9.81], device=self.device))
        accel_xy = proper_accel_b[:, 0:2] / 9.81
        self.prev_lin_vel = self.robot.data.root_lin_vel_w.clone()

        
        distance =  (self.forward_marker_locations - self.command_marker_locations)[:, :2]
        distance_norm = torch.norm((self.forward_marker_locations - self.command_marker_locations)[:, :2], dim=-1, keepdim=True)
        obs = torch.hstack((distance, distance_norm, gyro_z, accel_xy, dot, cross))
        # obs = torch.hstack((distance, dot, cross))
        # observations = {"policy": obs}
        
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)

        self.current_distance = torch.norm((self.forward_marker_locations - self.command_marker_locations)[:, :2], dim=-1, keepdim=True)

        if self.reset_flag:
            print("reset")
            self.last_distance = self.current_distance
            total_reward = torch.zeros((self.cfg.scene.num_envs)).cuda()
            self.reset_flag = False
        else:
            # distance_trend = self.last_distance - self.current_distance
            # distance_exp = torch.exp( - 1.0 * self.current_distance)
            # distance_reward = distance_trend*1000.0 + distance_exp*2.0
            # success_flag = torch.logical_and((self.current_distance < self.success_dist),(torch.norm(self.velocity, dim=-1, keepdim=True) <self.success_vel))
            # success_reward = success_flag.float() * 50.0
            # total_reward = (distance_reward*torch.logical_not(success_flag)) + (success_reward*self.success_bonus_flag) #+ alignment_reward
            
            progress_reward = self.last_distance - self.current_distance

            distance_exp = torch.exp( - 1.0 * self.current_distance)
            
            alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
            
            reach_wp_flag = (self.current_distance < self.reach_wp_dist)
            reach_wp_reward = reach_wp_flag.float() * 1.0

            success_flag = torch.logical_and((self.current_distance < self.success_dist),(torch.norm(self.velocity, dim=-1, keepdim=True) <self.success_vel))
            success_reward = success_flag.float() * 1.0
            total_reward = (
                1000.0 * progress_reward
                + 2.0 * distance_exp
                # + 1.0 * velocity_reward
                + 1.0 * alignment_reward
                # + 0.1 * spaciousness_reward lidar
                + 10.0 * reach_wp_reward*self.reach_wp_bonus_flag
                + 100.0 * success_reward*self.success_bonus_flag
                # - 10.0 * proximity_penalty
                # - 2.0 * spin_penalty
                # - 0.5 * steer_jerk_penalty
                # - 5.0 * stall_penalty
                # - 50.0 * crashed
                # - 5.0 * self.task_failed
            )
            
            self.success_buffer =  self.success_buffer + (self.current_distance < self.success_dist)
            self.success_bonus_flag = torch.logical_and(self.success_bonus_flag, torch.logical_not(success_flag))
            self.reach_wp_buffer =  self.reach_wp_buffer + (self.current_distance < self.reach_wp_dist)
            self.reach_wp_bonus_flag = torch.logical_and(self.reach_wp_bonus_flag, torch.logical_not(reach_wp_flag))
            self.last_distance = self.current_distance
        return total_reward 



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        #时间到达了
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print("Reach WP Number: ", self.cfg.scene.num_envs-self.reach_wp_bonus_flag.sum().item())
        print("Success Arrived Number: ", self.cfg.scene.num_envs-self.success_bonus_flag.sum().item())
        # if self.current_distance is not None: 
        #     mask = self.success_bonus_flag.bool()
        #     valid_distances = (self.current_distance[mask.squeeze()]).unsqueeze(1)
        #     print("Failed Car current distance: ", valid_distances)
        #     command_loc = self.marker_loc[mask.squeeze()].unsqueeze(1)
        #     print("Failed Car command location: ", command_loc)


        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)
        
        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5

        self.marker_loc = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_loc[:,-1] = 0.5
        self.marker_loc = self.marker_loc + self.env_origins

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()



        self.last_distance = None
        self.current_distance = None
        self.success_buffer=torch.zeros((self.cfg.scene.num_envs,1), device=self.device, dtype=torch.bool)
        self.success_bonus_flag = torch.ones((self.cfg.scene.num_envs,1), device=self.device, dtype=torch.bool)
        self.reach_wp_buffer=torch.zeros((self.cfg.scene.num_envs,1), device=self.device, dtype=torch.bool)
        self.reach_wp_bonus_flag = torch.ones((self.cfg.scene.num_envs,1), device=self.device, dtype=torch.bool)
        self.reset_flag = True
        self.task_failed[env_ids] = False
        self.prev_lin_vel[env_ids] = 0
