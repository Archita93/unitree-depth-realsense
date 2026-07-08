from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def raycaster_depth_norm(env, sensor_cfg: SceneEntityCfg, max_distance: float, invert: bool = True):
    sensor = env.scene.sensors[sensor_cfg.name]  # RayCasterCamera instance
    depth = sensor.data.output["distance_to_image_plane"]  # (num_envs, H, W, 1)
    # safety: ensure finite
    depth = torch.nan_to_num(depth, nan=max_distance, posinf=max_distance, neginf=max_distance)
    depth = torch.clamp(depth, 0.0, max_distance)

    depth01 = depth / max_distance
    if invert:
        depth01 = 1.0 - depth01
    # flatten for MLP policy
    return depth01.view(depth01.shape[0], -1)

