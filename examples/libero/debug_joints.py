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


@dataclasses.dataclass
class Args:
    host: str = "172.20.1.100"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"  # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
    task_keyword: str = ""          # filter tasks whose description contains this string (case-insensitive). empty = first task
    num_steps_wait: int = 10
    num_trials: int = 1             # how many rollouts to run (was 50 per task)
    max_steps: int = 100            # cap per episode

    log_every: int = 1              # print joint info every N steps (1 = every step)
    save_csv: bool = False          # also save a CSV of joint data

    use_wandb: bool = True         # enable wandb logging
    wandb_project: str = "libero-debug"
    wandb_run_name: str = ""        # empty = auto-generated

    save_video: bool = True        # save rollout video (mp4)
    video_fps: int = 10

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
                              toppra_predictions=None):
    """x/y/z별로 시간축 그래프: 실제 EEF + chunk 예측 + TOPP-RA 오버레이.

    toppra_predictions : list of (start_step, eef_smooth)
                         eef_smooth = (M, 3) or None (toppra 미설치 시)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual = np.array(actual_eef)
    n_steps = len(actual)
    labels = ["x", "y", "z"]
    actual_colors = ["r", "g", "b"]
    chunk_colors  = ["orange", "limegreen", "deepskyblue"]
    toppra_colors = ["darkorange", "darkgreen", "dodgerblue"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for dim, (ax, label, ac, cc, tc) in enumerate(
        zip(axes, labels, actual_colors, chunk_colors, toppra_colors)
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

        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"EEF Trajectory over Time (trial {trial})\n"
                 "solid=chunk(IK)  dotted=TOPP-RA smoothed")
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

    # --- find matching task ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    selected_task = None
    selected_task_id = None
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
            name=args.wandb_run_name or None,
            config={**dataclasses.asdict(args), "task_description": task_description},
        )

    # --- build env ---
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / selected_task.problem_folder / selected_task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(42)

    robot = env.env.robots[0]
    print(f"\n=== Robot Info ===")
    print(f"  type        : {type(robot.robot_model).__name__}")
    print(f"  dof         : {robot.dof}")
    print(f"  torque_limits: {robot.torque_limits}")
    print(f"==================\n")

    initial_states = task_suite.get_task_init_states(selected_task_id)

    # --- connect to policy server ---
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    ctrl_freq = env.env.control_freq
    toppra_planner = JointToppraPlanner(controller_freq=ctrl_freq)

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
        obs = env.set_init_state(initial_states[trial % len(initial_states)])

        action_plan = collections.deque()
        frames = []  # for robot video
        chunk_frames = []  # for chunk visualization video
        actual_eef = []  # actual EEF positions per step
        chunk_predictions = []  # (start_step, start_pos, predicted_eef) per replan
        ik_records = []         # (start_step, x_traj, fk_check) per replan
        toppra_predictions = [] # (start_step, eef_smooth) per replan
        t = 0

        while t < args.max_steps + args.num_steps_wait:
            # warm-up: let objects settle
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            step = t - args.num_steps_wait  # real step index

            # 실제 EEF 위치 기록
            actual_eef.append(obs["robot0_eef_pos"].copy())

            # get & log joint data
            pos, vel, torque = _get_joint_data(env)

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
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
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

                action_plan.extend(action_chunk[:args.replan_steps])

                # Diff IK 검증: chunk EEF → joint → FK 복원
                kin = MujocoKinematics(env.env.robots[0])
                q0, x_traj, R_traj, dt = extract_diff_ik_inputs(obs, action_chunk, env.env.robots[0], ctrl_freq)
                if step == 0:
                    print_diff_ik_inputs(q0, x_traj, R_traj, dt)
                q_waypoints = diff_ik_trajectory(q0, x_traj, R_traj, dt, kin)
                fk_check = np.array([kin.fk_pos(q) for q in q_waypoints])
                ik_err = np.abs(fk_check - x_traj)
                print(f"  [IK err] max={ik_err.max(axis=0).round(4)}  mean={ik_err.mean(axis=0).round(4)}")
                ik_records.append((step, x_traj, fk_check))

                # joint velocity clamp: TOPP-RA 입력 전 한계 초과 waypoint 보정
                qdot = np.diff(q_waypoints, axis=0) * ctrl_freq        # (T-1, 7) rad/s
                scale = np.max(np.abs(qdot) / toppra_planner.vel_limits, axis=1, keepdims=True)  # (T-1, 1)
                scale = np.maximum(scale, 1.0)                          # 한계 이내는 그대로
                q_waypoints_clamped = q_waypoints.copy()
                for i in range(1, len(q_waypoints)):
                    q_waypoints_clamped[i] = q_waypoints_clamped[i - 1] + (
                        q_waypoints[i] - q_waypoints[i - 1]
                    ) / scale[i - 1, 0]

                # TOPP-RA: joint waypoints → velocity/accel-constrained smooth EEF trajectory
                if step == 0:
                    print(f"  [TOPP-RA] q_waypoints shape: {q_waypoints.shape}")
                    qdot = np.diff(q_waypoints, axis=0) * ctrl_freq  # rad/s
                    print(f"  [TOPP-RA] max joint vel: {np.abs(qdot).max(axis=0).round(3)}")
                    print(f"  [TOPP-RA] panda vel lim: {toppra_planner.vel_limits.round(3)}")
                eef_smooth = toppra_planner.plan(q_waypoints_clamped, fk_pos=kin.fk_pos, inv_dyn=kin.inv_dyn)
                if step == 0 and eef_smooth is not None:
                    print(f"  [TOPP-RA] eef_smooth shape: {eef_smooth.shape}  (원본 T={q_waypoints.shape[0]})")
                toppra_predictions.append((step, eef_smooth))

            action = action_plan.popleft()
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

        if chunk_frames:
            import imageio
            chunk_video_path = pathlib.Path("videos") / f"chunk_trial{trial}.mp4"
            chunk_video_path.parent.mkdir(exist_ok=True)
            imageio.mimwrite(str(chunk_video_path), chunk_frames, fps=2)
            print(f"  Saved chunk video: {chunk_video_path.resolve()}")
            if args.use_wandb:
                wandb.log({f"video/chunk_trial{trial}": wandb.Video(str(chunk_video_path), fps=2, format="mp4")})

        if actual_eef:
            save_3d_trajectory_plot(actual_eef, chunk_predictions, trial, args.use_wandb)
            save_time_trajectory_plot(actual_eef, chunk_predictions, trial, args.use_wandb,
                                      toppra_predictions=toppra_predictions)
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
