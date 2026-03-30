"""
Quick debug script: run a single LIBERO task with a specific language instruction,
logging joint pos / vel / torque at every step.

Usage:
    python examples/libero/debug_joints.py \
        --task-keyword "pick up" \
        --num-trials 1 \
        --max-steps 50

The script connects to a running serve_policy.py server (default: localhost:8000).
"""

import collections
import dataclasses
import logging
import math
import pathlib
from typing import Literal

import numpy as np
import tyro
import wandb
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robot_kinematics import (
    MujocoKinematics,
    diff_ik_trajectory,
    extract_diff_ik_inputs,
    print_diff_ik_inputs,
    save_ik_verification_plot,
)
from toppra_viz import JointToppraPlanner

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


# ── Per-robot joint limits ─────────────────────────────────────────────────────
@dataclasses.dataclass
class RobotConfig:
    dof: int
    vel_limits:    np.ndarray   # rad/s
    acc_limits:    np.ndarray   # rad/s²
    torque_limits: np.ndarray   # N·m
    friction_coef: np.ndarray   # viscous friction (N·m·s/rad)

ROBOT_CONFIGS: dict = {
    "Panda": RobotConfig(
        dof=7,
        vel_limits    = np.array([2.175, 2.175, 2.175, 2.175, 2.61,  2.61,  2.61 ]),
        acc_limits    = np.array([15.0,  7.5,   10.0,  12.5,  15.0,  20.0,  20.0 ]),
        torque_limits = np.array([87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0 ]),
        friction_coef = np.array([0.2,   0.2,   0.15,  0.15,  0.1,   0.1,   0.1  ]),
    ),
    "UR5e": RobotConfig(
        dof=6,
        vel_limits    = np.array([2.094, 2.094, 2.094, 3.14,  3.14,  3.14  ]),
        acc_limits    = np.array([1.57,  1.57,  1.57,  1.57,  1.57,  1.57  ]),
        torque_limits = np.array([150.0, 150.0, 150.0, 28.0,  28.0,  28.0  ]),
        friction_coef = np.array([0.2,   0.2,   0.15,  0.1,   0.1,   0.1   ]),
    ),
    "IIWA": RobotConfig(
        dof=7,
        vel_limits    = np.array([1.484, 1.484, 1.745, 1.745, 2.269, 2.269, 2.269]),
        acc_limits    = np.array([20.0,  20.0,  20.0,  20.0,  20.0,  20.0,  20.0  ]),
        torque_limits = np.array([176.0, 176.0, 110.0, 110.0, 110.0, 40.0,  40.0 ]),
        friction_coef = np.array([0.2,   0.2,   0.15,  0.15,  0.1,   0.1,   0.1  ]),
    ),
    "Sawyer": RobotConfig(
        dof=7,
        vel_limits    = np.array([1.74,  1.74,  1.74,  1.74,  3.49,  3.49,  3.49 ]),
        acc_limits    = np.array([5.0,   5.0,   5.0,   5.0,   8.0,   8.0,   8.0  ]),
        torque_limits = np.array([80.0,  80.0,  80.0,  80.0,  20.0,  20.0,  20.0 ]),
        friction_coef = np.array([0.2,   0.2,   0.15,  0.15,  0.1,   0.1,   0.1  ]),
    ),
    "Kinova3": RobotConfig(
        dof=7,
        vel_limits    = np.array([1.396, 1.396, 1.396, 1.396, 1.222, 1.222, 1.222]),
        acc_limits    = np.array([5.0,   5.0,   5.0,   5.0,   5.0,   5.0,   5.0  ]),
        torque_limits = np.array([39.0,  39.0,  39.0,  39.0,  9.0,   9.0,   9.0  ]),
        friction_coef = np.array([0.2,   0.2,   0.15,  0.15,  0.1,   0.1,   0.1  ]),
    ),
}


@dataclasses.dataclass
class Args:
    host: str = "10.1.1.25"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5 # 5

    # ── choose robot ───────────────────────────────────────────────────────────────
    # Panda | IIWA 
    robot_name: str = "Panda"

    task_suite_name: str = "libero_spatial"  # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
    task_keyword: str = ""          # filter tasks whose description contains this string (case-insensitive). empty = first task
    task_index: int = 0            # task index (-1 = use task_keyword or first task)
    num_steps_wait: int = 10
    num_trials: int = 1             # how many rollouts to run (was 50 per task)
    max_steps: int = 100            # cap per episode

    # ── 실행 모드 ──────────────────────────────────────────────────────────────
    # osc_chunk    : 원래 방식 — policy chunk를 OSC로 그대로 실행 (TOPP-RA는 시각화만)
    # osc_toppra   : TOPP-RA 결과를 delta EEF로 변환해 OSC로 실행
    # toppra_global: TOPP-RA + QuadraticAlphaSurrogate로 끝 속도 스케일업
    execution_mode: Literal["osc_chunk", "osc_toppra", "toppra_global"] = "osc_chunk"

    # ── toppra_global img_limits (baseline rollout에서 관찰된 최대 절댓값) ──────
    
    # Measured physical limits (from rollout)
    # img_vel_limits: tuple = (0.071408, 0.487535, 0.104895, 0.479261, 0.154833, 0.305744, 0.303806)  # measured baseline
    # img_torque_limits: tuple = (4.685017, 61.12016, 4.828411, 23.30906, 2.879619, 4.859772, 2.376167)  # measured baseline
    
    # 2/3 of Panda hardware limits (PANDA_VEL_LIMITS * 2/3, PANDA_TORQUE_LIMITS * 2/3)
    img_vel_limits: tuple = (1.45, 1.45, 1.45, 1.45, 1.74, 1.74, 1.74)        # rad/s per joint
    img_torque_limits: tuple = (58.0, 58.0, 58.0, 58.0, 8.0, 8.0, 8.0)        # Nm per joint
    
    # original pandas spec
    # img_vel_limits: tuple = (2.175, 2.175, 2.175, 2.175, 2.61,  2.61,  2.61)        # rad/s per joint
    # img_torque_limits: tuple = (87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0 )        # Nm per joint
    
    alpha_star_max: float = 2        # toppra_global: alpha* 상한값

    # ── OSC controller ─────────────────────────────────────────────────────────
    osc_kp: float = 2000.0           # OSC position+orientation Kp gain (default: 2000 from osc_pose.json)
    osc_ramp_ratio: float = 1.0      # OSC ramp ratio (default: 0.2 from osc_pose.json, 1.0=no ramp)

    log_every: int = 1              # print joint info every N steps (1 = every step)
    save_csv: bool = False          # also save a CSV of joint data

    use_wandb: bool = True         # enable wandb logging
    wandb_project: str = "libero-SpeedUp"
    wandb_run_name: str = ""        # empty = auto-generated

    save_video: bool = True        # save rollout video (mp4)
    video_fps: int = 10



OSC_POS_SCALE = 0.05  # osc_pose.json output_max xyz
OSC_ORI_SCALE = 0.5   # osc_pose.json output_max rot

# toppra_global img_limits bias (img_limits에 더해지는 여유값)
IMG_VEL_BIAS   = 0.5   # rad/s
IMG_TORQUE_BIAS = 5.0  # Nm

# LIBERO init_states는 Panda 기준으로 저장됨
_PANDA_NQ       = 48   # Panda 환경 qpos 총 크기
_PANDA_NV       = 43   # Panda 환경 qvel 총 크기
_PANDA_N_ROBOT  = 9    # Panda arm(7) + PandaGripper(2)


def _adapt_init_state(panda_state: np.ndarray, env) -> np.ndarray: # for Cross-Embodiment exp
    """Panda 기준으로 저장된 init_state를 현재 로봇 환경에 맞게 변환.

    로봇 관절은 현재 로봇의 init_qpos(arm) + 0(gripper)로 채우고,
    오브젝트 state는 저장된 값 그대로 사용.
    """
    sim = env.env.sim

    # 저장된 state가 이미 현재 환경과 맞으면 그대로 반환
    if len(panda_state) == sim.model.nq + sim.model.nv:
        return panda_state

    # Panda state 파싱 (state[0]=time, state[1:1+nq]=qpos, state[1+nq:]=qvel)
    _OFFSET = 1  # time field
    panda_qpos = panda_state[_OFFSET : _OFFSET + _PANDA_NQ]
    panda_qvel = panda_state[_OFFSET + _PANDA_NQ : _OFFSET + _PANDA_NQ + _PANDA_NV]
    obj_qpos   = panda_qpos[_PANDA_N_ROBOT:]
    obj_qvel   = panda_qvel[_PANDA_N_ROBOT:]

    # 현재 로봇의 robot joint 수 계산 (arm + gripper, 모두 1-DOF)
    joint_names = list(sim.model.joint_names)
    n_robot = sum(1 for j in joint_names if j.startswith("robot0_") or j.startswith("gripper0_"))

    nq = sim.model.nq
    nv = sim.model.nv

    # 새 state 구성
    new_qpos = np.zeros(nq)
    new_qpos[n_robot:] = obj_qpos

    # arm init_qpos 채우기 (gripper는 0 유지)
    arm_qpos = env.env.robots[0].robot_model.init_qpos
    n_arm = len(arm_qpos)
    new_qpos[:n_arm] = arm_qpos

    new_qvel = np.zeros(nv)
    new_qvel[n_robot:] = obj_qvel

    result = np.concatenate([[panda_state[0]], new_qpos, new_qvel])  # time + qpos + qvel
    return result


def _build_osc_targets_from_toppra(q_smooth, action_chunk, kin, n_steps):
    """TOPP-RA joint trajectory → absolute EEF target tuples.

    실행 시점에 delta = target_pos - actual_pos 로 계산하여
    tracking error가 매 스텝 보정되도록 한다.

    q_smooth    : (M, N) TOPP-RA smoothed joint positions
    action_chunk: (T, 7) policy 출력 (gripper 값 참조용)
    kin         : MujocoKinematics
    n_steps     : 몇 스텝치 target을 만들지

    Returns
    -------
    list of (target_pos (3,), delta_ori (3,), gripper float)
    """
    targets = []
    for i in range(min(n_steps, len(q_smooth) - 1)):
        target_pos = kin.fk_pos(q_smooth[i + 1])
        orig_i     = min(i, len(action_chunk) - 1)
        delta_ori  = action_chunk[orig_i, 3:6]
        gripper    = float(action_chunk[orig_i, 6])
        targets.append((target_pos, delta_ori, gripper))
    return targets


def predict_eef_trajectory(robot, obs, action_chunk):
    """action_chunk (N, 7) OSC actions → predicted EEF positions (N, 3)"""
    current_pos = obs["robot0_eef_pos"].copy()
    predicted = []
    for action in action_chunk:
        scaled = robot.controller.scale_action(action[:6])
        current_pos = current_pos + scaled[:3]
        predicted.append(current_pos.copy())
    return np.array(predicted)  # (N, 3)


def project_points(points_3d, sim, cam_name, img_w, img_h):
    """3D world 좌표 → 2D 이미지 픽셀 좌표 (MuJoCo 카메라 투영)."""
    cam_id = sim.model.camera_name2id(cam_name)
    cam_pos = np.array(sim.data.cam_xpos[cam_id])
    cam_rot = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    fov_y = sim.model.cam_fovy[cam_id]  # degrees
    f = img_h / (2.0 * np.tan(np.deg2rad(fov_y) / 2.0))

    pixels = []
    for p in points_3d:
        p_cam = cam_rot.T @ (np.array(p) - cam_pos)
        depth = -p_cam[2]
        if depth <= 0:
            pixels.append(None)
            continue
        u = int(img_w - 1 - (f * p_cam[0] / depth + img_w / 2))
        v = int(-f * p_cam[1] / depth + img_h / 2)
        pixels.append((u, v))
    return pixels


def render_chunk_overlay(robot_img, predicted_eef, current_eef, sim, cam_name, step):
    """로봇 카메라 이미지 위에 predicted EEF trajectory를 오버레이."""
    import cv2
    # 이미지가 RGB이므로 BGR로 변환 후 그리고 다시 RGB로 변환
    img = cv2.cvtColor(robot_img.copy(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    all_points = [current_eef] + list(predicted_eef)
    pixels = project_points(all_points, sim, cam_name, w, h)

    # 선 그리기 (초록→빨강, BGR)
    for i in range(len(pixels) - 1):
        if pixels[i] is None or pixels[i + 1] is None:
            continue
        t = i / max(len(pixels) - 2, 1)
        color = (0, int(255 * (1 - t)), int(255 * t))  # BGR: 초록→빨강
        cv2.line(img, pixels[i], pixels[i + 1], color, 2)

    # 점 그리기
    for i, px in enumerate(pixels):
        if px is None:
            continue
        if i == 0:
            cv2.circle(img, px, 6, (0, 255, 255), -1)  # BGR yellow = 현재 위치
        else:
            t = (i - 1) / max(len(pixels) - 2, 1)
            color = (0, int(255 * (1 - t)), int(255 * t))
            cv2.circle(img, px, 3, color, -1)

    cv2.putText(img, f"step {step}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_time_trajectory_plot(actual_eef, chunk_predictions, trial, use_wandb,
                              toppra_predictions=None, toppra_global_predictions=None):
    """x/y/z별로 시간축 그래프: 실제 EEF + chunk 예측 + TOPP-RA 오버레이.

    toppra_predictions        : list of (start_step, eef_smooth)
    toppra_global_predictions : list of (start_step, eef_smooth_tg)
                                eef_smooth = (M, 3) or None (toppra 미설치 시)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual = np.array(actual_eef)
    n_steps = len(actual)
    labels = ["x", "y", "z"]
    actual_colors  = ["r", "g", "b"]
    chunk_colors   = ["orange", "limegreen", "deepskyblue"]
    toppra_colors  = ["darkorange", "darkgreen", "dodgerblue"]
    tg_colors      = ["red", "forestgreen", "royalblue"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for dim, (ax, label, ac, cc, tc, tgc) in enumerate(
        zip(axes, labels, actual_colors, chunk_colors, toppra_colors, tg_colors)
    ):
        # ── 실제 EEF ──────────────────────────────────────────────────────────
        ax.plot(range(n_steps), actual[:, dim], color=ac, linewidth=2, label="actual")

        # ── IK-based chunk 예측 ───────────────────────────────────────────────
        for i, (start_step, _, chunk) in enumerate(chunk_predictions):
            t_range = list(range(start_step, start_step + len(chunk)))
            ax.plot(t_range, chunk[:, dim], color=cc, alpha=0.5, linewidth=1,
                    linestyle="-", label="chunk" if i == 0 else None)

            if i + 1 < len(chunk_predictions):
                next_step, next_pos, _ = chunk_predictions[i + 1]
                idx = next_step - start_step
                if 0 <= idx < len(chunk):
                    ax.scatter(next_step, chunk[idx, dim], color=cc, s=30, marker="x", zorder=5,
                               label="chunk pred at replan" if i == 0 else None)
                ax.scatter(next_step, next_pos[dim], color=ac, s=40, zorder=5,
                           label="actual at replan" if i == 0 else None)

        # ── TOPP-RA smoothed (dotted) ─────────────────────────────────────────
        if toppra_predictions:
            for i, (start_step, eef_smooth) in enumerate(toppra_predictions):
                if eef_smooth is None:
                    continue
                t_range = list(range(start_step, start_step + len(eef_smooth)))
                ax.plot(t_range, eef_smooth[:, dim], color=tc, alpha=0.8, linewidth=1.5,
                        linestyle=":", label="toppra" if i == 0 else None)

        # ── TOPP-RA global (dashed) ───────────────────────────────────────────
        if toppra_global_predictions:
            for i, (start_step, eef_smooth_tg) in enumerate(toppra_global_predictions):
                if eef_smooth_tg is None:
                    continue
                t_range = list(range(start_step, start_step + len(eef_smooth_tg)))
                ax.plot(t_range, eef_smooth_tg[:, dim], color=tgc, alpha=0.9, linewidth=1.5,
                        linestyle="--", label="toppra_global" if i == 0 else None)

        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"EEF Trajectory over Time (trial {trial})\n"
                 "solid=chunk(IK)  dotted=TOPP-RA  dashed=TOPP-RA global")
    fig.tight_layout()

    out_path = pathlib.Path("videos") / f"traj_time_trial{trial}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved time trajectory: {out_path.resolve()}")

    if use_wandb:
        wandb.log({f"traj_time/trial{trial}": wandb.Image(str(out_path))})


def save_3d_trajectory_plot(actual_eef, chunk_predictions, trial, use_wandb):
    """실제 EEF 경로 + replan마다 chunk 예측을 3D로 겹쳐 그리기."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401 — registers 3d projection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 실제 경로
    actual = np.array(actual_eef)
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "k-", linewidth=2, label="actual", zorder=5)
    ax.scatter(*actual[0], color="green", s=80, zorder=6, label="start")
    ax.scatter(*actual[-1], color="red", s=80, zorder=6, label="end")

    # chunk 예측들
    for i, (_, start_pos, chunk) in enumerate(chunk_predictions):
        pts = np.vstack([start_pos, chunk])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.4, linewidth=1,
                color=plt.cm.cool(i / max(len(chunk_predictions) - 1, 1)))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"EEF Trajectory (trial {trial})")
    ax.legend()

    out_path = pathlib.Path("videos") / f"traj3d_trial{trial}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3D trajectory: {out_path.resolve()}")

    if use_wandb:
        wandb.log({f"traj3d/trial{trial}": wandb.Image(str(out_path))})


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_joint_data(env: OffScreenRenderEnv):
    """Extract joint pos, vel, torque from the underlying robosuite robot."""
    robot = env.env.robots[0]
    pos    = np.array(robot._joint_positions)          # 7-dim
    vel    = np.array(robot._joint_velocities)         # 7-dim
    torque = np.array(robot.torques) if robot.torques is not None else np.zeros(len(pos))  # 7-dim
    return pos, vel, torque


def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("toppra").setLevel(logging.WARNING)
    logging.getLogger("robot_kinematics").setLevel(logging.WARNING)

    # --- find matching task ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    selected_task = None
    selected_task_id = None
    if args.task_index >= 0:
        selected_task = task_suite.get_task(args.task_index)
        selected_task_id = args.task_index
    else:
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if args.task_keyword.lower() in task.language.lower():
                selected_task = task
                selected_task_id = task_id
                break

    if selected_task is None:
        if args.task_keyword:
            raise ValueError(f"No task found matching '{args.task_keyword}' in suite '{args.task_suite_name}'")
        selected_task = task_suite.get_task(0)
        selected_task_id = 0

    task_description = selected_task.language
    print(f"\n=== Task: {task_description} ===\n")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.execution_mode}_{int(args.osc_kp)}",
            config={**dataclasses.asdict(args), "task_description": task_description},
        )

    # --- validate robot ---
    if args.robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot '{args.robot_name}'. Available: {list(ROBOT_CONFIGS)}")
    robot_cfg = ROBOT_CONFIGS[args.robot_name]

    # --- build env ---
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / selected_task.problem_folder / selected_task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        robots=[args.robot_name],
    )
    env.seed(42)

    robot = env.env.robots[0]
    print(f"\n=== Robot Info ===")
    print(f"  name        : {args.robot_name}")
    print(f"  type        : {type(robot.robot_model).__name__}")
    print(f"  dof         : {robot.dof}")
    print(f"  vel_limits  : {robot_cfg.vel_limits}")
    print(f"  torque_limits: {robot_cfg.torque_limits}")
    print(f"  nq (qpos size): {env.env.sim.model.nq}")
    print(f"  nv (qvel size): {env.env.sim.model.nv}")
    print(f"  joint_names   : {list(env.env.sim.model.joint_names)}")
    print(f"==================\n")


    initial_states = task_suite.get_task_init_states(selected_task_id)

    dummy_action = list(LIBERO_DUMMY_ACTION)

    # --- connect to policy server ---
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    ctrl_freq = env.env.control_freq
    toppra_planner = JointToppraPlanner(
        controller_freq=ctrl_freq,
        vel_limits=robot_cfg.vel_limits,
        acc_limits=robot_cfg.acc_limits,
        torque_limits=robot_cfg.torque_limits,
        friction_coef=robot_cfg.friction_coef,
    )

    joint_names = [f"j{i}" for i in range(7)]
    header = (
        "step,"
        + ",".join(f"pos_{n}" for n in joint_names) + ","
        + ",".join(f"vel_{n}" for n in joint_names) + ","
        + ",".join(f"trq_{n}" for n in joint_names)
    )

    csv_rows = [header]

    for trial in range(args.num_trials):
        print(f"\n--- Trial {trial+1}/{args.num_trials} ---")
        env.reset()

        # Override OSC controller params after reset (hard_reset recreates controller from json)
        _ctrl = env.env.robots[0].controller
        _ctrl.kp = np.full(6, args.osc_kp)
        _ctrl.kd = 2 * np.sqrt(_ctrl.kp)
        _ctrl.ramp_ratio = args.osc_ramp_ratio

        # ── 액추에이터 목록 출력 (gripper 관련 확인용) ──────────────────────────
        sim = env.env.sim
        print("[actuators]")
        for _act_id in range(sim.model.nu):
            _act_name = sim.model.actuator_id2name(_act_id)
            print(f"  {_act_id}: {_act_name}")

        obs = env.set_init_state(_adapt_init_state(initial_states[trial % len(initial_states)], env))
        # 오브젝트 위치 확인
        for k, v in obs.items():
            if "pos" in k and "robot" not in k:
                print(f"  {k}: {v}")
        print(f"  robot0_eef_pos: {obs['robot0_eef_pos']}")

        action_plan = collections.deque()
        frames = []  # for robot video
        side_frames = []  # for side-view video
        chunk_frames = []  # for chunk visualization video
        actual_eef = []  # actual EEF positions per step
        chunk_predictions = []  # (start_step, start_pos, predicted_eef) per replan
        ik_records = []         # (start_step, x_traj, fk_check) per replan
        toppra_predictions = [] # (start_step, eef_smooth) per replan
        toppra_global_predictions = [] # (start_step, eef_smooth_tg) per replan (toppra_global mode only)
        all_vels    = []        # (step, 7) joint velocities
        all_torques = []        # (step, 7) joint torques
        t = 0

        while t < args.max_steps + args.num_steps_wait:
            # warm-up: let objects settle
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(dummy_action)
                # warm-up 매 스텝마다 arm을 초기 위치로 고정 (오브젝트만 settling)
                _sim = env.env.sim
                _robot = env.env.robots[0]
                for i, idx in enumerate(_robot._ref_joint_pos_indexes):
                    _sim.data.qpos[idx] = _robot.robot_model.init_qpos[i]
                    _sim.data.qvel[idx] = 0.0
                _sim.forward()
                t += 1
                continue

            step = t - args.num_steps_wait  # real step index

            # 실제 EEF 위치 기록
            actual_eef.append(obs["robot0_eef_pos"].copy())

            # get & log joint data
            pos, vel, torque = _get_joint_data(env)
            all_vels.append(vel.copy())
            all_torques.append(torque.copy())

            if step % args.log_every == 0:
                if args.use_wandb:
                    log_dict = {f"trial": trial, "step": step, "task": task_description}
                    for i, (p, v, trq) in enumerate(zip(pos, vel, torque)):
                        log_dict[f"pos/j{i}"] = p
                        log_dict[f"vel/j{i}"] = v
                        log_dict[f"torque/j{i}"] = trq
                    wandb.log(log_dict, step=trial * args.max_steps + step)

            if args.save_csv:
                csv_rows.append(
                    f"{step},"
                    + ",".join(f"{v:.6f}" for v in pos) + ","
                    + ",".join(f"{v:.6f}" for v in vel) + ","
                    + ",".join(f"{v:.6f}" for v in torque)
                )

            # collect frame for video (raw resolution before resize)
            if args.save_video:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                _sim = env.env.sim
                _cid = _sim.model.camera_name2id("agentview")
                _op, _oq = _sim.model.cam_pos[_cid].copy(), _sim.model.cam_quat[_cid].copy()
                _sim.model.cam_pos[_cid]  = np.array([-0.1, 1.5, 1.3])
                _sim.model.cam_quat[_cid] = np.array([0.707, -0.707, 0.0, 0.0])
                _sim.forward()
                _sf = _sim.render(camera_name="agentview", height=LIBERO_ENV_RESOLUTION, width=LIBERO_ENV_RESOLUTION)
                _sim.model.cam_pos[_cid], _sim.model.cam_quat[_cid] = _op, _oq
                _sim.forward()  # cam_xpos/xmat 복원 (project_points가 이 값을 읽음)
                side_frames.append(np.ascontiguousarray(_sf))

            # query policy
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            if not action_plan:
                # gripper obs: Panda는 [0.04(open)~0(closed)] x2,
                # Robotiq85는 finger_joint [0(open)~0.8(closed)] → Panda 스케일로 변환
                _gqpos = obs["robot0_gripper_qpos"]
                if len(_gqpos) == 2:
                    gripper_obs = _gqpos  # Panda
                else:
                    _v = float(_gqpos[0])
                    _norm = (1.0 - _v / 0.8) * 0.04
                    gripper_obs = np.array([_norm, _norm])

                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        gripper_obs,
                    )),
                    "prompt": str(task_description),
                }

                # chunk : delta EEF position
                action_chunk = client.infer(element)["actions"]
                predicted_eef = predict_eef_trajectory(robot, obs, action_chunk)
                chunk_predictions.append((step, obs["robot0_eef_pos"].copy(), predicted_eef))
                robot_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                chunk_frame = render_chunk_overlay(robot_img, predicted_eef, obs["robot0_eef_pos"], env.env.sim, "agentview", step)
                chunk_frames.append(chunk_frame)

                # ── Diff IK: chunk EEF → joint waypoints ──────────────────────
                kin = MujocoKinematics(env.env.robots[0])
                q0, x_traj, R_traj, dt = extract_diff_ik_inputs(obs, action_chunk, env.env.robots[0], ctrl_freq)
                q_waypoints = diff_ik_trajectory(q0, x_traj, R_traj, dt, kin)

                fk_check = np.array([kin.fk_pos(q) for q in q_waypoints])
                ik_records.append((step, x_traj, fk_check))

                # ── joint velocity clamp ───────────────────────────────────────
                qdot = np.diff(q_waypoints, axis=0) * ctrl_freq
                scale = np.maximum(np.max(np.abs(qdot) / toppra_planner.vel_limits, axis=1, keepdims=True), 1.0)
                q_waypoints_clamped = q_waypoints.copy()
                for i in range(1, len(q_waypoints)):
                    q_waypoints_clamped[i] = q_waypoints_clamped[i - 1] + (
                        q_waypoints[i] - q_waypoints[i - 1]
                    ) / scale[i - 1, 0]

                # ── TOPP-RA: 항상 계산 (시각화 + 선택적 실행) ─────────────────
                if step == 0:
                    print("[TOPP-RA] torque constraint: ON (inv_dyn=kin.inv_dyn)")
                eef_smooth, q_smooth = toppra_planner.plan(
                    q_waypoints_clamped, fk_pos=kin.fk_pos, inv_dyn=kin.inv_dyn,
                )
                toppra_predictions.append((step, eef_smooth))

                # ── execution_mode에 따라 action_plan 채우기 ───────────────────
                if args.execution_mode == "osc_chunk":
                    # 원래 방식: policy chunk 그대로 실행
                    action_plan.extend(action_chunk[:args.replan_steps])

                elif args.execution_mode == "osc_toppra":
                    # TOPP-RA → delta EEF → OSC 실행
                    if q_smooth is not None:
                        toppra_osc = _build_osc_targets_from_toppra(
                            q_smooth, action_chunk, kin, args.replan_steps
                        )
                        action_plan.extend(toppra_osc if toppra_osc else action_chunk[:args.replan_steps])
                    else:
                        action_plan.extend(action_chunk[:args.replan_steps])  # fallback


                elif args.execution_mode == "toppra_global":
                    # TOPP-RA + QuadraticAlphaSurrogate로 끝 속도 스케일업
                    from speedup_finalspeed import QuadraticAlphaSurrogate, dls_qdot
                    q0_cur = np.array(env.env.robots[0]._joint_positions)
                    Jv = kin.jacobian(q0_cur)[:3]   # (3, N)
                    M  = kin.mass_matrix(q0_cur)    # (N, N)
                    p  = obs["robot0_eef_pos"].copy()

                    # chunk 첫 EEF delta → v0 (task-space velocity)
                    v0 = action_chunk[0, :3] * OSC_POS_SCALE * ctrl_freq  # (3,)
                    if np.linalg.norm(v0) < 1e-6:
                        v0 = np.array([1e-4, 0.0, 0.0])
                    qdot0 = dls_qdot(Jv, v0)

                    # h_samples, Jvdotqdot_samples at alpha=0,1,2
                    h_samples, jdot_samples = {}, {}
                    for alpha in [0.0, 1.0, 2.0]:
                        h_samples[alpha] = kin.bias_forces(q0_cur, alpha * qdot0)
                        jdot_samples[alpha] = kin.jac_dot_qdot(q0_cur, alpha * qdot0)

                    img_vel = np.asarray(args.img_vel_limits, dtype=float) + IMG_VEL_BIAS
                    img_tau = np.asarray(args.img_torque_limits, dtype=float) + IMG_TORQUE_BIAS
                    img_limits = {
                        "dq_min": -img_vel, "dq_max": img_vel,
                        "tau_min": -img_tau, "tau_max": img_tau,
                    }
                    tar_limits = {
                        "dq_min": -toppra_planner.vel_limits, "dq_max": toppra_planner.vel_limits,
                        "tau_min": -toppra_planner.torque_limits, "tau_max": toppra_planner.torque_limits,
                    }

                    surrogate = QuadraticAlphaSurrogate(
                        Jv=Jv, M=M, p=p, v0=v0, qdot0=qdot0,
                        h_samples=h_samples, Jvdotqdot_samples=jdot_samples,
                        img_limits=img_limits, tar_limits=tar_limits,
                        dt= 1*(1.0 / ctrl_freq),
                    )
                    result = surrogate.compute_alpha_star()
                    alpha_star = min(result["alpha_star"], args.alpha_star_max)
                    if step % args.log_every == 0:
                        print(f"[toppra_global] step={step}  alpha*={alpha_star:.3f} (raw={result['alpha_star']:.3f})  feasible={result['feasible']}")

                    # chunk 끝 joint velocity 스케일업
                    qdot_end_base = (q_waypoints_clamped[-1] - q_waypoints_clamped[-2]) * ctrl_freq
                    qdot_end_scaled = alpha_star * qdot_end_base

                    # wandb: alpha* 및 관절별 끝 속도 비교 로그
                    if args.use_wandb:
                        log_dict = {
                            "toppra_global/alpha_star": alpha_star,
                            "toppra_global/feasible": float(result["feasible"]),
                            "toppra_global/alpha_vel_bound": result.get("alpha_vel_bound", float("nan")),
                            "toppra_global/qdot_end_base_norm": float(np.linalg.norm(qdot_end_base)),
                            "toppra_global/qdot_end_scaled_norm": float(np.linalg.norm(qdot_end_scaled)),
                        }
                        for j in range(len(qdot_end_base)):
                            log_dict[f"toppra_global/qdot_end_base_j{j}"]   = float(qdot_end_base[j])
                            log_dict[f"toppra_global/qdot_end_scaled_j{j}"] = float(qdot_end_scaled[j])
                        wandb.log(log_dict, step=step)

                    result_tg = toppra_planner.plan(
                        q_waypoints_clamped, fk_pos=kin.fk_pos, inv_dyn=kin.inv_dyn,
                        qdot_end_override=qdot_end_scaled,
                    )
                    if result_tg is None or result_tg[0] is None:
                        print(f"[toppra_global] step={step}  TOPP-RA FAILED → fallback to plain toppra")
                        if args.use_wandb:
                            wandb.log({"toppra_global/toppra_failed": 1}, step=step)
                        # fallback: alpha 없는 첫 번째 TOPP-RA 결과 사용
                        if q_smooth is not None:
                            toppra_osc = _build_osc_targets_from_toppra(
                                q_smooth, action_chunk, kin, args.replan_steps
                            )
                            action_plan.extend(toppra_osc if toppra_osc else action_chunk[:args.replan_steps])
                        else:
                            action_plan.extend(action_chunk[:args.replan_steps])
                    else:
                        eef_smooth_tg, q_smooth_tg = result_tg
                        toppra_global_predictions.append((step, eef_smooth_tg))
                        if args.use_wandb:
                            wandb.log({"toppra_global/toppra_failed": 0}, step=step)
                        toppra_osc = _build_osc_targets_from_toppra(
                            q_smooth_tg, action_chunk, kin, args.replan_steps
                        )
                        action_plan.extend(toppra_osc if toppra_osc else action_chunk[:args.replan_steps])

            item = action_plan.popleft()
            if isinstance(item, tuple):
                # toppra absolute target: compute delta from current actual pos
                target_pos, delta_ori, gripper = item
                delta_pos = (target_pos - obs["robot0_eef_pos"]) / OSC_POS_SCALE
                action = np.concatenate([delta_pos, delta_ori, [gripper]])
            else:
                action = item.copy()
            action[6] = 1.0 if action[6] > 0 else -1.0  # binary gripper
            obs, _, done, _ = env.step(action.tolist())
            t += 1

            if done:
                print(f"  >> SUCCESS at step {step}")
                if args.use_wandb:
                    wandb.log({"result/success": 1, "result/steps_to_success": step, "trial": trial})
                break
        else:
            if args.use_wandb:
                wandb.log({"result/success": 0, "result/steps_to_success": args.max_steps, "trial": trial})

        if args.save_video and frames:
            import imageio
            video_path = pathlib.Path("videos") / f"rollout_trial{trial}.mp4"
            video_path.parent.mkdir(exist_ok=True)
            imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
            print(f"  Saved video: {video_path.resolve()}")
            if args.use_wandb:
                wandb.log({f"video/trial{trial}": wandb.Video(str(video_path), fps=args.video_fps, format="mp4")})

        if args.save_video and side_frames:
            import imageio
            side_video_path = pathlib.Path("videos") / f"sideview_trial{trial}.mp4"
            side_video_path.parent.mkdir(exist_ok=True)
            imageio.mimwrite(str(side_video_path), side_frames, fps=args.video_fps)
            print(f"  Saved side video: {side_video_path.resolve()}")
            if args.use_wandb:
                wandb.log({f"video/side_trial{trial}": wandb.Video(str(side_video_path), fps=args.video_fps, format="mp4")})

        if chunk_frames:
            import imageio
            chunk_video_path = pathlib.Path("videos") / f"chunk_trial{trial}.mp4"
            chunk_video_path.parent.mkdir(exist_ok=True)
            imageio.mimwrite(str(chunk_video_path), chunk_frames, fps=2)
            print(f"  Saved chunk video: {chunk_video_path.resolve()}")
            if args.use_wandb:
                wandb.log({f"video/chunk_trial{trial}": wandb.Video(str(chunk_video_path), fps=2, format="mp4")})

        if args.use_wandb and all_vels:
            vels    = np.array(all_vels)    # (T, 7)
            torques = np.array(all_torques) # (T, 7)
            cols = ["joint", "vel_mean", "vel_max", "vel_min", "vel_absmax",
                    "trq_mean", "trq_max", "trq_min", "trq_absmax"]
            table = wandb.Table(columns=cols)
            for i in range(vels.shape[1]):
                table.add_data(
                    f"j{i}",
                    float(np.mean(vels[:, i])),
                    float(np.max(vels[:, i])),
                    float(np.min(vels[:, i])),
                    float(np.max(np.abs(vels[:, i]))),
                    float(np.mean(torques[:, i])),
                    float(np.max(torques[:, i])),
                    float(np.min(torques[:, i])),
                    float(np.max(np.abs(torques[:, i]))),
                )
            wandb.log({f"joint_stats/trial{trial}": table})

        if actual_eef:
            save_3d_trajectory_plot(actual_eef, chunk_predictions, trial, args.use_wandb)
            save_time_trajectory_plot(actual_eef, chunk_predictions, trial, args.use_wandb,
                                      toppra_predictions=toppra_predictions,
                                      toppra_global_predictions=toppra_global_predictions)
            if ik_records:
                save_ik_verification_plot(ik_records, trial, use_wandb=args.use_wandb)

    env.close()

    if args.use_wandb:
        wandb.finish()

    if args.save_csv:
        out = pathlib.Path("csv") / "joint_log.csv"
        out.parent.mkdir(exist_ok=True)
        out.write_text("\n".join(csv_rows))
        print(f"\nSaved joint log to {out.resolve()}")


if __name__ == "__main__":
    tyro.cli(main)
