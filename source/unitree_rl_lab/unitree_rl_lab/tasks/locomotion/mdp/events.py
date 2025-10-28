from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def random_camera_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    pos_noise_range: dict[str,tuple[float,float]] | None = None,
    rot_noise_range: dict[str,tuple[float,float]] | None = None,
    convention: str = 'ros',
):
    """
    prestartup
    """
    camera_sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    init_rot = torch.tensor(camera_sensor.cfg.offset.rot).repeat(env.num_envs,1).to(env.device)

    if pos_noise_range is not None: 
        pos_range_list = [pos_noise_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        pos_ranges = torch.tensor(pos_range_list, device=env.device)
        random_pose = math_utils.sample_uniform(pos_ranges[:,0], pos_ranges[:,1], (env.num_envs,1), device=env.device)
    else:
        random_pose = None
    if rot_noise_range is not None:
        rot_range_list = [rot_noise_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
        rot_ranges = torch.deg2rad(torch.tensor(rot_range_list)).to(env.device)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(init_rot)
        init_rot = torch.stack([roll, pitch, yaw], dim=-1).to(env.device)
        init_rot += math_utils.sample_uniform(rot_ranges[:,0], rot_ranges[:,1], (env.num_envs,1), device=env.device)
        random_rot = math_utils.quat_from_euler_xyz(init_rot[:,0],init_rot[:,1],init_rot[:,2])
    else:
        random_rot = init_rot 

    camera_sensor.set_world_poses(
        positions=random_pose,
        orientations=random_rot,
        convention=convention,
        env_ids=torch.arange(env.num_envs, dtype=torch.int64, device=env.device),
    )
