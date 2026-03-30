from pathlib import Path

import isaaclab.sim as sim_utils
from .RFI_PDActuatorCfg import RFI_PDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _resolve_g1_usd_path() -> str:
    """Resolve the G1 USD path independent of current working directory."""
    usd_rel_path = Path("g1_29dof_rev_1_0") / "g1_29dof_rev_1_0.usd"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / usd_rel_path
        if candidate.is_file():
            return str(candidate)
    # Fallback keeps previous behavior if repository layout changes.
    return str(usd_rel_path)


G1_DEX_FIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_resolve_g1_usd_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,  
            # 29
            "waist_pitch_joint":0.0,
            "waist_roll_joint":0.0,
            "waist_yaw_joint":0.0,
            'left_shoulder_pitch_joint': 0.23,
            'left_shoulder_roll_joint': 0.25,  
            'left_shoulder_yaw_joint': 0.0,   
            'left_elbow_joint': 0.9, # 0.9
            'left_wrist_roll_joint': 0.0,     
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,      
            'right_shoulder_pitch_joint': 0.23,
            'right_shoulder_roll_joint': -0.25, 
            'right_shoulder_yaw_joint': 0.0,  
            'right_elbow_joint': 0.9, # 0.9
            'right_wrist_roll_joint': 0.0,    
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0, 
            # finger
            # ".*_proximal_joint":0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": RFI_PDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                "waist_yaw_joint": 100,
                "waist_roll_joint": 100,
                "waist_pitch_joint": 100,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
                "waist_yaw_joint": 2.0,
                "waist_roll_joint": 2.0,
                "waist_pitch_joint": 2.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_.*": 0.01,
            },
            rfi=(-0.01, 0.01),
            min_delay=3,
            max_delay=7,
        ),
        "feet": RFI_PDActuatorCfg(
            effort_limit=50,
            velocity_limit=37,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            rfi=(-0.01, 0.01),
            min_delay=3,
            max_delay=7,
        ),
        "arms": RFI_PDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 50.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_roll_joint": 30.0,
                ".*_wrist_pitch_joint": 30.0,
                ".*_wrist_yaw_joint": 30.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
                ".*_wrist_yaw_joint": 2.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
            rfi=(-0.01, 0.01),
            min_delay=3,
            max_delay=7,
        ),
    },
)
G1_SHOE = G1_DEX_FIX.copy()
G1_SHOE.spawn.usd_path = "./g1_29dof_rev_3_0/g1_29dof_rev_3_0.usd"
G1_SHOE.init_state.pos = (0.0, 0.0, 0.81)