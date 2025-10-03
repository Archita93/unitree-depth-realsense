import math
import scipy.spatial.transform as tf

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, RayCasterCameraCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion import mdp

def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())

IMAGE_W = 48
IMAGE_H = 48
CAMERA_TRANSFORM_TRANS = (0.3, 0.0, 0.3)
CAMERA_TRANSFORM_ROT = tf.Rotation.from_euler("xzy", (90.0, -90.0, 60.0), degrees=True).as_quat()[[3, 0, 1, 2]].tolist()
# CAMERA_TRANSFORM_ROT = (0.627211, 0.326506, -0.326506, -0.627211)

DEBUG_VIS = True

PARKOUR_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            gap_width_range=(0.15, 0.8),
            platform_width=1.5,
            proportion=0.1,
        ),
        "ring": terrain_gen.MeshFloatingRingTerrainCfg(
            ring_width_range=(0.1, 1.0),
            ring_height_range=(0.25, 0.35),
            ring_thickness=0.5,
            platform_width=1.5,
            proportion=0.1,
        ),
        "box": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(0.05, 0.5),
            platform_width=1.5,
            double_box=True,
            proportion=0.1,
        ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            pit_depth_range=(0.05, 0.5),
            platform_width=1.5,
            double_pit=True,
            proportion=0.1,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #     stone_width_range=(0.05, 0.05),
        #     stone_distance_range=(0.05, 0.05),
        #     stone_height_max=0.1,
        #     platform_width=2.0,
        #     proportion=0.1,
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },
)

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=PARKOUR_TERRAIN_CFG,
        # terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=DEBUG_VIS,
        mesh_prim_paths=["/World/ground"],
    )
    roof_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0], direction=[0.0, 0.0, 1.0]),
        debug_vis=DEBUG_VIS,
        mesh_prim_paths=["/World/ground"],
    )

    # rgb_camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/mounted_camera",
    #
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=CAMERA_TRANSFORM_TRANS,
    #         rot=CAMERA_TRANSFORM_ROT,
    #         convention="opengl",
    #     ),
    #     update_period=0.1,
    #     debug_vis=DEBUG_VIS,
    #     width=IMAGE_W,
    #     height=IMAGE_H,
    #     spawn=sim_utils.PinholeCameraCfg(),
    #     data_types=[
    #         "rgb",
    #         "distance_to_image_plane",
    #     ],
    # )

    depth_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",

        offset=RayCasterCameraCfg.OffsetCfg(
            pos=CAMERA_TRANSFORM_TRANS,
            # rot=(0.5, 0.5, -0.5, -0.5),
            rot=CAMERA_TRANSFORM_ROT,
            # rot=quat_from_euler_rpy(90.0, -90.0, 0.0, degrees=True),
            convention="opengl",
        ),
        update_period=0.1,
        ray_alignment="base",
        max_distance=1.0,
        depth_clipping_behavior="max",
        debug_vis=DEBUG_VIS,
        # debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            width=IMAGE_W,
            height=IMAGE_H
        ),
        mesh_prim_paths=["/World/ground"],
        data_types=[
            "distance_to_image_plane",
        ],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformLevelVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.1,
    #     debug_vis=DEBUG_VIS,
    #     ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
    #     ),
    #     limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
    #     ),
    # )

    base_velocity = mdp.UniformPose2dVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        simple_heading=True,
        debug_vis=DEBUG_VIS,
        stop_distance=0.2,
        min_distance=8.0,
        ranges=mdp.UniformPose2dVelocityCommandCfg.Ranges(
            lin_vel=(-0.2, 0.2),
            ang_vel=(-0.4, 0.4),
            pos_x=(-4, 4),
            pos_y=(-4, 4),
            heading=(-1, 1)
        ),
        limit_ranges=mdp.UniformPose2dVelocityCommandCfg.Ranges(
            lin_vel=(-1.0, 1.0),
            ang_vel=(-0.4, 0.4),
            pos_x=(-4, 4),
            pos_y=(-4, 4),
            heading=(-1, 1)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class VisionCfg(ObsGroup):
        image = ObsTerm(func=mdp.image, params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "data_type": "distance_to_image_plane",
            "normalize": True,
        })
        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    vision: VisionCfg = VisionCfg()

    # @configclass
    # class ColorVisionCfg(ObsGroup):
    #     image = ObsTerm(func=mdp.image, params={
    #         "sensor_cfg": SceneEntityCfg("rgb_camera"),
    #         "data_type": "rgb",
    #         "normalize": False,
    #     })
    #     def __post_init__(self):
    #         # self.history_length = 5
    #         self.enable_corruption = False
    #         self.concatenate_terms = True
    #
    # color_vision: ColorVisionCfg = ColorVisionCfg()


    @configclass
    class StudentCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: StudentCfg = StudentCfg()

    @configclass
    class TeacherCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        roof_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("roof_scanner")},
            clip=(-1.0, 5.0),
        )

        # def __post_init__(self):
        #     self.history_length = 5

    # privileged observations
    critic: TeacherCfg = TeacherCfg()

    student: StudentCfg | None = None
    teacher: TeacherCfg | None = None


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    # goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": "base_velocity", "distance": 0.3})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_goal)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels_goal)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # custom config
        self.scene.robot.actuators["legs"].stiffness = 40.0
        self.scene.robot.actuators["legs"].damping = 1.0


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        # self.scene.terrain.terrain_generator.num_cols = len(COBBLESTONE_ROAD_CFG.sub_terrains)
        self.scene.terrain.terrain_generator.num_cols = len(PARKOUR_TERRAIN_CFG.sub_terrains)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

@configclass
class TeacherEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # compute only one observation cfg
        self.scene.depth_camera = None
        self.observations.vision = None
        self.observations.critic = None
        self.observations.policy = self.observations.TeacherCfg()
        self.observations.teacher = None
        self.observations.student = None

@configclass
class TeacherPlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # compute only one observation cfg
        self.scene.depth_camera = None
        self.observations.vision = None
        self.observations.critic = None
        self.observations.policy = self.observations.TeacherCfg()
        self.observations.teacher = None
        self.observations.student = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

@configclass
class RecordEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # compute only one observation cfg
        self.observations.vision = self.observations.VisionCfg()
        self.observations.critic = None
        self.observations.policy = self.observations.TeacherCfg()
        self.observations.teacher = self.observations.TeacherCfg()
        self.observations.student = self.observations.StudentCfg()
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RecordPlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # compute only one observation cfg
        self.observations.vision = self.observations.VisionCfg()
        self.observations.critic = None
        self.observations.policy = self.observations.TeacherCfg()
        self.observations.teacher = self.observations.TeacherCfg()
        self.observations.student = self.observations.StudentCfg()
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class StudentEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # compute only one observation cfg
        self.observations.vision = self.observations.VisionCfg()
        self.observations.critic = self.observations.TeacherCfg()
        self.observations.policy = self.observations.StudentCfg()
        self.observations.teacher = None
        self.observations.student = None

@configclass
class StudentPlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # compute only one observation cfg
        self.observations.vision = self.observations.VisionCfg()
        self.observations.critic = self.observations.TeacherCfg()
        self.observations.policy = self.observations.StudentCfg()
        self.observations.teacher = None
        self.observations.student = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

