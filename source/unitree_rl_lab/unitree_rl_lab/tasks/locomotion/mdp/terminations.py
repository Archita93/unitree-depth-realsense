from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm
    from .commands import UniformPose2dVelocityCommand

"""
MDP terminations.
"""

def goal_reached(env: ManagerBasedRLEnv, command_name: str, distance: float) -> torch.Tensor:
    command: UniformPose2dVelocityCommand = env.command_manager.get_term(command_name)
    return command.distances < distance
