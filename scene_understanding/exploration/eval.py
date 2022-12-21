#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import asnumpy, rearrange

from habitat import Config, logger
from habitat_baselines.common.utils import batch_obs, generate_video
from habitat_extensions.utils import observations_to_image

from common.env_utils import construct_envs
from common.environments import ExpRLEnv
from common.default import get_config

from models.mapnet import DepthProjectionNet
from models.occant import OccupancyAnticipator

from policy_utils import OccupancyAnticipationWrapper

from utils.common import (add_pose, convert_gt2channel_to_gtrgb,
                                 convert_world2map)
from utils.metrics import (TemporalMetric,
                                  measure_area_seen_performance,
                                  measure_map_quality,
                                  measure_pose_estimation_performance)
from utils.visualization import generate_topdown_allocentric_map

from occupancy_anticipation import ActiveNeuralSLAMExplorer

import sys
ROOT_DIR = "/home/fedynyak/votenet"
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls

from object_detection.votenet import votenet_build, votenet_detection, votenet_nms
votenet = votenet_build()

class OccAntExpTrainer:
    r"""Trainer class for Occupancy Anticipated based exploration algorithm.
    """
    def __init__(self, config=None):
        if config is not None:
            self._synchronize_configs(config)
        self.config = config

        # Set pytorch random seed for initialization
        torch.manual_seed(config.PYT_RANDOM_SEED)

        self.mapper = None
        self.local_actor_critic = None
        self.global_actor_critic = None
        self.ans_net = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

    def _synchronize_configs(self, config):
        config.defrost()
        config.RL.ANS.PLANNER.nplanners = config.NUM_PROCESSES
        config.RL.ANS.MAPPER.thresh_explored = config.RL.ANS.thresh_explored
        config.RL.ANS.pyt_random_seed = config.PYT_RANDOM_SEED
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.pyt_random_seed = config.PYT_RANDOM_SEED

        # Compute the EGO_PROJECTION options based on the
        # depth sensor information and agent parameters.
        map_size = config.RL.ANS.MAPPER.map_size
        map_scale = config.RL.ANS.MAPPER.map_scale
        min_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        hfov_rad = np.radians(float(hfov))
        vfov_rad = 2 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
        vfov = np.degrees(vfov_rad).item()
        camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        height_thresholds = [0.2, 1.5]

        # Set the EGO_PROJECTION options
        ego_proj_config = config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION
        ego_proj_config.local_map_shape = (2, map_size, map_size)
        ego_proj_config.map_scale = map_scale
        ego_proj_config.min_depth = min_depth
        ego_proj_config.max_depth = max_depth
        ego_proj_config.hfov = hfov
        ego_proj_config.vfov = vfov
        ego_proj_config.camera_height = camera_height
        ego_proj_config.height_thresholds = height_thresholds
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION = ego_proj_config

        # Set the GT anticipation options
        wall_fov = config.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.WALL_FOV = wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SIZE = map_size
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SCALE = map_scale
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAX_SENSOR_RANGE = -1

        # Set the correct image scaling values
        config.RL.ANS.MAPPER.image_scale_hw = config.RL.ANS.image_scale_hw
        config.RL.ANS.LOCAL_POLICY.image_scale_hw = config.RL.ANS.image_scale_hw

        # Set the agent dynamics for the local policy
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.forward_step = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE

        if "COLLISION_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
            config.TASK_CONFIG.TASK.SENSORS.append("COLLISION_SENSOR")
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_EXP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")

        config.freeze()

    def _setup_actor_critic_agent(self, ans_cfg: Config):

        occ_cfg = ans_cfg.OCCUPANCY_ANTICIPATOR
        mapper_cfg = ans_cfg.MAPPER

        # Create occupancy anticipation model
        occupancy_model = OccupancyAnticipator(occ_cfg)
        occupancy_model = OccupancyAnticipationWrapper(occupancy_model, mapper_cfg.map_size, (128, 128))

        # Create ANS model
        self.ans_net = ActiveNeuralSLAMExplorer(ans_cfg, occupancy_model)
        self.mapper = self.ans_net.mapper
        self.local_actor_critic = self.ans_net.local_policy
        self.global_actor_critic = self.ans_net.global_policy
        
        # Create depth projection model to estimate visible occupancy
        self.depth_projection_net = DepthProjectionNet(ans_cfg.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION)
        
        # Set to device
        self.mapper.to(self.device)
        self.local_actor_critic.to(self.device)
        self.global_actor_critic.to(self.device)
        self.depth_projection_net.to(self.device)

    def _convert_actions_to_delta(self, actions):
        """actions -> torch Tensor
        """
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        delta_xyt = torch.zeros(1, 3, device=self.device)
        # Forward step
        act_mask = actions.squeeze(1) == 0
        delta_xyt[act_mask, 0] = sim_cfg.FORWARD_STEP_SIZE
        # Turn left
        act_mask = actions.squeeze(1) == 1
        delta_xyt[act_mask, 2] = math.radians(-sim_cfg.TURN_ANGLE)
        # Turn right
        act_mask = actions.squeeze(1) == 2
        delta_xyt[act_mask, 2] = math.radians(sim_cfg.TURN_ANGLE)
        return delta_xyt

    def _prepare_batch(self, observations, prev_batch=None, device=None, actions=None):
        imH, imW = self.config.RL.ANS.image_scale_hw
        device = self.device if device is None else device
        batch = batch_obs(observations, device=device)
        if batch["rgb"].size(1) != imH or batch["rgb"].size(2) != imW:
            rgb = rearrange(batch["rgb"], "b h w c -> b c h w")
            rgb = F.interpolate(rgb, (imH, imW), mode="bilinear")
            batch["rgb"] = rearrange(rgb, "b c h w -> b h w c")
        if batch["depth"].size(1) != imH or batch["depth"].size(2) != imW:
            depth = rearrange(batch["depth"], "b h w c -> b c h w")
            depth = F.interpolate(depth, (imH, imW), mode="nearest")
            batch["depth"] = rearrange(depth, "b c h w -> b h w c")

        # Compute ego_map_gt from depth
        ego_map_gt_b = self.depth_projection_net(rearrange(batch["depth"], "b h w c -> b c h w"))
        batch["ego_map_gt"] = rearrange(ego_map_gt_b, "b c h w -> b h w c")
        
        if actions is None:
            # Initialization condition
            # If pose estimates are not available, set the initial estimate to zeros.
            if "pose" not in batch:
                # Set initial pose estimate to zero
                batch["pose"] = torch.zeros(1, 3).to(self.device)
            batch["prev_actions"] = torch.zeros(1, 1).to(self.device)
        else:
            # Rollouts condition
            # If pose estimates are not available, compute them from action taken.
            if "pose" not in batch:
                assert prev_batch is not None
                actions_delta = self._convert_actions_to_delta(actions)
                batch["pose"] = add_pose(prev_batch["pose"], actions_delta)
            batch["prev_actions"] = actions

        return batch

    def _eval_checkpoint(self, checkpoint_path):

        self.device = torch.device("cuda", self.config.TORCH_GPU_ID)

        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")

        config = self.config.clone()

        ans_cfg = config.RL.ANS

        logger.info(f"env config: {config}")

        self.envs = construct_envs(config, ExpRLEnv)
        self._setup_actor_critic_agent(ans_cfg)

        ckpt_dict["mapper_state_dict"] = {k[7:]: v for k, v in ckpt_dict["mapper_state_dict"].items() if k.startswith("mapper.")}
        self.mapper.load_state_dict(ckpt_dict["mapper_state_dict"])
        ckpt_dict["local_state_dict"] = {k[13:]: v for k, v in ckpt_dict["local_state_dict"].items() if k.startswith("actor_critic.")}
        self.local_actor_critic.load_state_dict(ckpt_dict["local_state_dict"])
        ckpt_dict["global_state_dict"] = {k[13:]: v for k, v in ckpt_dict["global_state_dict"].items() if k.startswith("actor_critic.")}
        self.global_actor_critic.load_state_dict(ckpt_dict["global_state_dict"])

        # Set models to evaluation
        self.mapper.eval()
        self.local_actor_critic.eval()
        self.global_actor_critic.eval()

        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        M = ans_cfg.overall_map_size
        s = ans_cfg.MAPPER.map_scale

        mapping_metrics = defaultdict(lambda: TemporalMetric())
        pose_estimation_metrics = defaultdict(lambda: TemporalMetric())

        observations = [self.envs.reset()]
        batch = self._prepare_batch(observations)
        prev_batch = batch
        state_estimates = {
            "pose_estimates": torch.zeros(1, 3).to(self.device),
            "map_states": torch.zeros(1, 2, M, M).to(self.device),
            "recurrent_hidden_states": torch.zeros(1, 1, ans_cfg.LOCAL_POLICY.hidden_size).to(self.device),
            "visited_states": torch.zeros(1, 1, M, M).to(self.device),
        }
        ground_truth_states = {
            "visible_occupancy": torch.zeros(1, 2, M, M).to(self.device),
            "pose": torch.zeros(1, 3).to(self.device),
            "environment_layout": None,
        }

        self.ans_net.reset()
        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)

        # Visualization stuff
        gt_agent_poses_over_time = [[]]
        pred_agent_poses_over_time = [[]]
        rgb_frames = [[]]

        gt_map_agent = asnumpy(convert_world2map(ground_truth_states["pose"], (M, M), s))
        pred_map_agent = asnumpy(convert_world2map(state_estimates["pose_estimates"], (M, M), s))
        pred_map_agent = np.concatenate([pred_map_agent, asnumpy(state_estimates["pose_estimates"][:, 2:3]),], axis=1)

        gt_agent_poses_over_time[0].append(gt_map_agent[0])
        pred_agent_poses_over_time[0].append(pred_map_agent[0])

        observations[0]["depth_fullsize"] = observations[0]["depth"]
        observations[0]["depth"] = cv2.resize(observations[0]["depth"], (128, 128))[:,:,None]

        observations[0]["rgb_fullsize"] = observations[0]["rgb"]
        observations[0]["rgb"] = cv2.resize(observations[0]["rgb"], (128, 128))

        for ep_step in range(self.config.T_EXP):
            ep_time = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device).fill_(ep_step)

            print(ep_step)
            # print(state_estimates["pose_estimates"])

            hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV * (np.pi / 180)
            W = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            H = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
            fx = W / np.tan(hfov / 2.) / 2
            fy = fx
            K = np.array([
                [fx, 0., W / 2, 0.],
                [0., fy, H / 2, 0.],
                [0., 0., 1, 0],
                [0., 0., 0, 1]])

            obs = {
                "image_depth": observations[0]["depth_fullsize"][...,0] * config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
                "image_depth_info": {
                    "width": W,
                    "height": H,
                    "matrix_intrinsics": K
                },
            }

            x, y, phi = state_estimates["pose_estimates"][0].detach().cpu().numpy()
            T1 = np.array([
                [np.cos(phi), 0, -np.sin(phi), x],
                [0, 0, 0, 0],
                [np.sin(phi), 0, np.cos(phi), y],
                [0, 0, 0, 1]])
            T2 = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 1.5],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

            results = votenet_detection(votenet, obs, T1, T2)
            print(f"Detected {len(results)} objects: {','.join(r['class'] for r in results)}")

            # from PIL import Image
            # if len(results) > 0:
            #     Image.fromarray(observations[0]["rgb_fullsize"]).save(f"images/{ep_step}.png")

            # if len(results) > 0:
            import matplotlib.pyplot as plt
            f = plt.figure()
            plt.imshow(observations[0]["rgb_fullsize"])
            for result in results:
                # if result["confidence"] < 0.75:
                #     continue

            # result = results[0]
                box_corners_camera = result["box_corners_camera"]
                xs = box_corners_camera[:,0]
                ys = box_corners_camera[:,1]
                zs = box_corners_camera[:,2]

                zs[zs < 0.001] = 0.001

                xxs = fx * xs / zs + W / 2
                yys = fy * ys / zs + H / 2

                plt.scatter(xxs, yys, s=5, c="red")
                plt.plot(list(xxs[:4]) + [xxs[0]], list(yys[:4]) + [yys[0]], lw=1, color="red")
                plt.plot(list(xxs[4:]) + [xxs[4]], list(yys[4:]) + [yys[4]], lw=1, color="red")
                plt.plot(xxs[0::4], yys[0::4], lw=1, color="red")
                plt.plot(xxs[1::4], yys[1::4], lw=1, color="red")
                plt.plot(xxs[2::4], yys[2::4], lw=1, color="red")
                plt.plot(xxs[3::4], yys[3::4], lw=1, color="red")
            # plt.savefig(f"images/{ep_step}.png", dpi=200)
            plt.xlim(0, W)
            plt.ylim(H, 0)
            plt.savefig(f"images/{str(ep_step).rjust(5, '0')}_clip.png", dpi=200)

            with torch.no_grad():
                (
                    local_policy_outputs,
                    state_estimates,
                ) = self.ans_net.act(
                    batch,
                    prev_batch,
                    state_estimates,
                    ep_time,
                    deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
                )

                actions = local_policy_outputs["actions"]
                prev_actions.copy_(actions)

            # Update GT estimates at t = ep_step
            ground_truth_states["pose"] = batch["pose_gt"]
            ground_truth_states["visible_occupancy"] = self.ans_net.mapper.ext_register_map(
                ground_truth_states["visible_occupancy"],
                batch["ego_map_gt"].permute(0, 3, 1, 2),
                batch["pose_gt"],
            )

            # Visualization stuff
            gt_map_agent = asnumpy(convert_world2map(ground_truth_states["pose"], (M, M), s))
            gt_map_agent = np.concatenate([gt_map_agent, asnumpy(ground_truth_states["pose"][:, 2:3])], axis=1)
            
            pred_map_agent = asnumpy(convert_world2map(state_estimates["pose_estimates"], (M, M), s))
            pred_map_agent = np.concatenate([pred_map_agent, asnumpy(state_estimates["pose_estimates"][:, 2:3])], axis=1)

            gt_agent_poses_over_time[0].append(gt_map_agent[0])
            pred_agent_poses_over_time[0].append(pred_map_agent[0])

            outputs = self.envs.step(action=actions[0][0].item())
            observations, _, dones, infos = [[x] for x in outputs]

            observations[0]["depth_fullsize"] = observations[0]["depth"]
            observations[0]["depth"] = cv2.resize(observations[0]["depth"], (128, 128))[:,:,None]

            observations[0]["rgb_fullsize"] = observations[0]["rgb"]
            observations[0]["rgb"] = cv2.resize(observations[0]["rgb"], (128, 128))

            if ep_step == 0:
                environment_layout = np.stack([info["gt_global_map"] for info in infos], axis=0)  # (bs, M, M, 2)
                environment_layout = rearrange(environment_layout, "b h w c -> b c h w")  # (bs, 2, M, M)
                environment_layout = torch.Tensor(environment_layout).to(self.device)
                ground_truth_states["environment_layout"] = environment_layout

            prev_batch = batch
            batch = self._prepare_batch(observations, prev_batch, actions=actions)

            ########################################################

            if ep_step == 0 or (ep_step + 1) % 50 == 0:
                curr_all_metrics = {}

                # Compute accumulative pose estimation error
                pose_hat_final = state_estimates["pose_estimates"]  # (bs, 3)
                pose_gt_final = ground_truth_states["pose"]  # (bs, 3)
                curr_pose_estimation_metrics = measure_pose_estimation_performance(pose_hat_final, pose_gt_final, reduction="sum")
                for k, v in curr_pose_estimation_metrics.items():
                    pose_estimation_metrics[k].update(v, 1, ep_step)
                curr_all_metrics.update(curr_pose_estimation_metrics)

                # Compute map quality
                curr_map_quality_metrics = measure_map_quality(
                    state_estimates["map_states"],
                    ground_truth_states["environment_layout"],
                    s,
                    entropy_thresh=1.0,
                    reduction="sum",
                    apply_mask=True,
                )
                for k, v in curr_map_quality_metrics.items():
                    mapping_metrics[k].update(v, 1, ep_step)
                curr_all_metrics.update(curr_map_quality_metrics)

                # Compute area seen
                curr_area_seen_metrics = measure_area_seen_performance(ground_truth_states["visible_occupancy"], s, reduction="sum")
                for k, v in curr_area_seen_metrics.items():
                    mapping_metrics[k].update(v, 1, ep_step)
                curr_all_metrics.update(curr_area_seen_metrics)

            ################################################
            if len(self.config.VIDEO_OPTION) > 0:
                # episode ended
                if ep_step == self.config.T_EXP - 1:
                    video_metrics = {}
                    for k in ["area_seen", "mean_iou", "map_accuracy"]:
                        video_metrics[k] = curr_all_metrics[k]
                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[0],
                            episode_id=0,
                            checkpoint_idx=0,
                            metrics=video_metrics,
                        )

                    from habitat.utils.visualizations.utils import images_to_video
                    import glob
                    images = []
                    for path in sorted(glob.glob("images/*.png")):
                        img = cv2.imread(path)[:, :, [2, 1, 0]]
                        images.append(img)
                    images_to_video(images, ".", "test")

                # episode continues
                else:
                    # (HSTACK) FRAME = RGB + DEPTH + TOPDOWN
                    frame = observations_to_image(observations[0], infos[0], observation_size=300)
                    
                    # (HSTACK) FRAME = FRAME + EGO_MAP_GT
                    ego_map_gt_i = asnumpy(batch["ego_map_gt"][0])  # (2, H, W)
                    ego_map_gt_i = convert_gt2channel_to_gtrgb(ego_map_gt_i)
                    ego_map_gt_i = cv2.resize(ego_map_gt_i, (300, 300))
                    frame = np.concatenate([frame, ego_map_gt_i], axis=1)

                    # Generate ANS specific visualizations
                    environment_layout = asnumpy(ground_truth_states["environment_layout"][0])  # (2, H, W)
                    visible_occupancy = asnumpy(ground_truth_states["visible_occupancy"][0])  # (2, H, W)
                    anticipated_occupancy = asnumpy(state_estimates["map_states"][0])  # (2, H, W)
                    curr_gt_poses = gt_agent_poses_over_time[0]
                    curr_pred_poses = pred_agent_poses_over_time[0]

                    H = frame.shape[0]

                    # VISIBLE OCCUPANCY VISUAL
                    visible_occupancy_vis = generate_topdown_allocentric_map(
                        environment_layout,
                        visible_occupancy,
                        curr_gt_poses,
                        thresh_explored=ans_cfg.thresh_explored,
                        thresh_obstacle=ans_cfg.thresh_obstacle,
                    )
                    visible_occupancy_vis = cv2.resize(visible_occupancy_vis, (H, H))

                    # ANTICIPATED OCCUPANCY VISUAL
                    anticipated_occupancy_vis = generate_topdown_allocentric_map(
                        environment_layout,
                        anticipated_occupancy,
                        curr_pred_poses,
                        thresh_explored=ans_cfg.thresh_explored,
                        thresh_obstacle=ans_cfg.thresh_obstacle,
                    )
                    anticipated_occupancy_vis = cv2.resize(anticipated_occupancy_vis, (H, H))

                    # ACTION MAP VISUAL
                    anticipated_action_map = generate_topdown_allocentric_map(
                        environment_layout,
                        anticipated_occupancy,
                        curr_pred_poses,
                        zoom=False,
                        thresh_explored=ans_cfg.thresh_explored,
                        thresh_obstacle=ans_cfg.thresh_obstacle,
                    )
                    global_goals = self.ans_net.states["curr_global_goals"]
                    local_goals = self.ans_net.states["curr_local_goals"]
                    if global_goals is not None:
                        cX = int(global_goals[0, 0].item())
                        cY = int(global_goals[0, 1].item())
                        anticipated_action_map = cv2.circle(anticipated_action_map, (cX, cY), 10, (255, 0, 0), -1)
                    if local_goals is not None:
                        cX = int(local_goals[0, 0].item())
                        cY = int(local_goals[0, 1].item())
                        anticipated_action_map = cv2.circle(anticipated_action_map, (cX, cY), 10, (0, 255, 255), -1)
                    anticipated_action_map = cv2.resize(anticipated_action_map, (H, H))

                    # (HSTACK) MAPS = VISIBLE_OCC + ANTICIPATED_OCC + ACTIONS + ZEROS (BLACK)
                    maps_vis = np.concatenate(
                        [
                            visible_occupancy_vis,
                            anticipated_occupancy_vis,
                            anticipated_action_map,
                            np.zeros_like(anticipated_action_map),
                        ],
                        axis=1,
                    )

                    # (VSTACK) FRAME = FRAME + MAPS
                    frame = np.concatenate([frame, maps_vis], axis=0)
                    rgb_frames[0].append(frame)

        self.envs.close()


if __name__== "__main__":
    cfg = get_config(["../configs/ppo_exploration.yaml"])
    model = OccAntExpTrainer(cfg)
    model._eval_checkpoint("/home/fedynyak/OccupancyAnticipation/checkpoints/ckpt.10.pth")
