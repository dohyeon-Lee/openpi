"""
Robot kinematics + Differential IK for trajectory planning.

진입점:
    from robot_kinematics import MujocoKinematics, extract_diff_ik_inputs, diff_ik_trajectory

    kin = MujocoKinematics(robot)          # robot = env.env.robots[0]
    q0, x_traj, R_traj, dt = extract_diff_ik_inputs(obs, action_chunk, robot, ctrl_freq)
    q_traj = diff_ik_trajectory(q0, x_traj, R_traj, dt, kin)

[REAL ROBOT 교체 가이드]
    PinocchioKinematics(urdf_path, eef_frame_name) 로 교체하면
    diff_ik_trajectory / extract_diff_ik_inputs 수정 없이 재사용 가능.
"""

import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)

OSC_POS_SCALE = 0.05  # osc_pose.json output_max xyz
OSC_ORI_SCALE = 0.5   # osc_pose.json output_max rot


# ── 회전 유틸 ──────────────────────────────────────────────────────────────────

def quat2rotmat(quat: np.ndarray) -> np.ndarray:
    """xyzw quaternion → (3, 3) rotation matrix"""
    x, y, z, w = quat
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


def axisangle2rotmat(aa: np.ndarray) -> np.ndarray:
    """axis-angle (3,) → (3, 3) rotation matrix (Rodrigues)"""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def log_SO3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle (so3 vector)"""
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2 * np.sin(theta))
    return theta * w


# ── MuJoCo Kinematics ─────────────────────────────────────────────────────────

class MujocoKinematics:
    """
    MuJoCo 기반 FK / Jacobian.

    Parameters
    ----------
    robot : robosuite robot object (env.env.robots[0])
            joint_names, sim 모두 robot에서 직접 참조.
    """

    def __init__(self, robot):
        self._robot = robot

        sim = robot.sim
        self.eef_site_name = self._find_eef_site(sim)

        # joint name → qpos index
        self.joint_qpos_indices = np.array([
            sim.model.jnt_qposadr[sim.model.joint_name2id(jn)]
            for jn in robot.robot_joints
        ])
        self.n_joints = len(self.joint_qpos_indices)
        logger.info("MujocoKinematics: site=%s  n_joints=%d", self.eef_site_name, self.n_joints)

        self._inv_dyn_data = None       # lazy init (env.reset() 후 model 교체 대응)
        self._inv_dyn_model_ptr = None  # 현재 data가 어떤 model용인지 추적

    @staticmethod
    def _find_eef_site(sim) -> str:
        site_names = [sim.model.site_id2name(i) for i in range(sim.model.nsite)]
        for candidate in ["gripper0_grip_site", "robot0_grip_site", "robot0_eef_site"]:
            if candidate in site_names:
                return candidate
        grip_sites = [s for s in site_names if "grip" in s.lower()]
        if grip_sites:
            return grip_sites[0]
        raise RuntimeError(f"EEF site not found. Available: {site_names}")

    @property
    def sim(self):
        """env.reset() 시 robot.sim이 교체되므로 항상 최신 참조 반환."""
        return self._robot.sim

    def _set_q(self, q):
        self.sim.data.qpos[self.joint_qpos_indices] = q
        self.sim.forward()

    def fk_pos(self, q: np.ndarray) -> np.ndarray:
        """(N,) → EEF position (3,)"""
        saved = self.sim.data.qpos.copy()
        try:
            self._set_q(q)
            return self.sim.data.get_site_xpos(self.eef_site_name).copy()
        finally:
            self.sim.data.qpos[:] = saved
            self.sim.forward()

    def fk_rot(self, q: np.ndarray) -> np.ndarray:
        """(N,) → EEF rotation matrix (3, 3)"""
        saved = self.sim.data.qpos.copy()
        try:
            self._set_q(q)
            return self.sim.data.get_site_xmat(self.eef_site_name).copy().reshape(3, 3)
        finally:
            self.sim.data.qpos[:] = saved
            self.sim.forward()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """(N,) → 6D Jacobian (6, N)  — rows 0:3 pos, 3:6 rot"""
        saved = self.sim.data.qpos.copy()
        try:
            self._set_q(q)
            nv = self.sim.model.nv
            jacp = self.sim.data.get_site_jacp(self.eef_site_name).reshape(3, nv)
            jacr = self.sim.data.get_site_jacr(self.eef_site_name).reshape(3, nv)
            return np.vstack([jacp, jacr])[:, self.joint_qpos_indices]
        finally:
            self.sim.data.qpos[:] = saved
            self.sim.forward()

    def inv_dyn(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Inverse dynamics: (q, qd, qdd) → joint torques (N,)
        τ = M(q)q̈ + C(q,q̇)q̇ + g(q)

        전용 MjData를 사용해 메인 sim 상태를 건드리지 않음 (빠름).
        env.reset() 후 model이 바뀌면 자동으로 재생성.
        """
        import mujoco
        model_ptr = self.sim.model._model

        # model이 바뀌었으면 전용 data 재생성
        if model_ptr is not self._inv_dyn_model_ptr:
            self._inv_dyn_data = mujoco.MjData(model_ptr)
            self._inv_dyn_model_ptr = model_ptr

        idx = self.joint_qpos_indices
        d = self._inv_dyn_data
        d.qpos[idx] = q
        d.qvel[idx] = qd
        d.qacc[idx] = qdd
        mujoco.mj_inverse(model_ptr, d)
        return np.array(d.qfrc_inverse[idx])


# ── 입력 추출 ─────────────────────────────────────────────────────────────────

def extract_diff_ik_inputs(obs: dict, action_chunk: np.ndarray, robot, ctrl_freq: float):
    """
    debug_joints.py에서 얻은 obs / action_chunk / robot으로
    diff IK에 필요한 초기 조건을 구성.

    Parameters
    ----------
    obs          : env step 반환 observation dict
    action_chunk : (T, 7) policy 출력  [Δpos/0.05, Δori/0.5, gripper]
    robot        : env.env.robots[0]
    ctrl_freq    : env.env.control_freq

    Returns
    -------
    q0     : (N,)      현재 joint positions
    x_traj : (T, 3)    목표 EEF position 시퀀스
    R_traj : (T, 3, 3) 목표 EEF rotation 시퀀스
    dt     : float     1 / ctrl_freq
    """
    q0 = np.array(robot._joint_positions)
    dt = 1.0 / ctrl_freq

    curr_pos = obs["robot0_eef_pos"].copy()
    curr_R   = quat2rotmat(obs["robot0_eef_quat"])

    x_list = [curr_pos.copy()]
    R_list = [curr_R.copy()]

    for action in action_chunk:
        curr_pos = curr_pos + action[:3] * OSC_POS_SCALE
        curr_R   = axisangle2rotmat(action[3:6] * OSC_ORI_SCALE) @ curr_R
        x_list.append(curr_pos.copy())
        R_list.append(curr_R.copy())

    # x_list[0] = 현재 상태 → [1:] 로 T개 target
    return q0, np.array(x_list[1:]), np.array(R_list[1:]), dt


def print_diff_ik_inputs(q0, x_traj, R_traj, dt):
    print("\n=== Diff IK Inputs ===")
    print(f"dt           : {dt:.4f}s  ({1/dt:.1f} Hz)")
    print(f"q0           : {np.round(q0, 4)}")
    print(f"x_traj[0]    : {np.round(x_traj[0], 4)}")
    print(f"x_traj[-1]   : {np.round(x_traj[-1], 4)}")
    print(f"x_traj shape : {x_traj.shape}")
    print(f"R_traj[0]    :\n{np.round(R_traj[0], 4)}")
    deltas = np.linalg.norm(np.diff(x_traj, axis=0), axis=1)
    print(f"Δpos/step(m) : min={deltas.min():.4f}  max={deltas.max():.4f}  mean={deltas.mean():.4f}")
    print("======================\n")


# ── Differential IK ───────────────────────────────────────────────────────────

def _damped_pinv(J: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    return J.T @ np.linalg.inv(J @ J.T + lam * np.eye(J.shape[0]))


def diff_ik_trajectory(
    q0: np.ndarray,
    x_traj: np.ndarray,
    R_traj: np.ndarray,
    dt: float,
    kin: MujocoKinematics,
    lam: float = 0.05,
    w_pos: float = 1.0,
    w_rot: float = 0.01,
    max_iters: int = 3,
    pos_tol: float = 1e-3,   # 1mm
) -> np.ndarray:
    """
    Differential IK trajectory tracking (DLS + error feedback, iterative per waypoint).

    Parameters
    ----------
    q0       : (N,)      초기 joint positions
    x_traj   : (T, 3)    목표 EEF position 시퀀스
    R_traj   : (T, 3, 3) 목표 EEF rotation 시퀀스
    dt       : float     timestep (velocity 스케일링용)
    kin      : MujocoKinematics
    max_iters: 각 waypoint당 최대 IK 반복 (inner loop)
    pos_tol  : position 수렴 기준 (m)

    Returns
    -------
    q_traj : (T, N) joint position trajectory
    """
    T = x_traj.shape[0]
    N = q0.shape[0]
    q = q0.copy()
    q_traj = np.zeros((T, N))
    q_traj[0] = q

    for t in range(T - 1):
        x_target = x_traj[t + 1]
        R_target = R_traj[t + 1]

        for _ in range(max_iters):
            x_curr = kin.fk_pos(q)
            R_curr = kin.fk_rot(q)

            pos_err = x_target - x_curr
            if np.linalg.norm(pos_err) < pos_tol:
                break

            v     = pos_err / dt
            omega = log_SO3(R_target @ R_curr.T) / dt
            v6    = np.concatenate([w_pos * v, w_rot * omega])
            J     = kin.jacobian(q)
            qdot  = _damped_pinv(J, lam) @ v6
            q     = q + qdot * dt

        q_traj[t + 1] = q

    return q_traj


# ── IK Verification Plot ───────────────────────────────────────────────────────

def save_ik_verification_plot(ik_records, trial, out_dir="videos", use_wandb=False):
    """chunk EEF (x_traj) vs IK→FK 복원 EEF (fk_check) 비교 그래프.

    Parameters
    ----------
    ik_records : list of (start_step, x_traj, fk_check)
        x_traj   : (T, 3) Cartesian 전파 타겟
        fk_check : (T, 3) diff IK → FK 복원값
    trial      : int     trial 번호 (파일명/wandb key에 사용)
    out_dir    : str     출력 디렉터리 (default: "videos")
    use_wandb  : bool    wandb에 이미지 업로드 여부

    Usage
    -----
    from robot_kinematics import save_ik_verification_plot
    save_ik_verification_plot(ik_records, trial, use_wandb=args.use_wandb)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i, (start_step, x_traj, fk_check) in enumerate(ik_records):
        t_range = list(range(start_step, start_step + len(x_traj)))
        for dim, ax in enumerate(axes):
            ax.plot(t_range, x_traj[:, dim],   color="orange",      alpha=0.8, lw=1.5,
                    linestyle="-",  label="chunk (x_traj)" if i == 0 else None)
            ax.plot(t_range, fk_check[:, dim], color="deepskyblue", alpha=0.8, lw=1.5,
                    linestyle="--", label="IK→FK" if i == 0 else None)

    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"IK Verification: chunk EEF vs IK→FK recovered EEF (trial {trial})\n"
                 "두 선이 겹칠수록 IK 정확도 높음")
    fig.tight_layout()

    out_path = pathlib.Path(out_dir) / f"ik_verify_trial{trial}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved IK verification: %s", out_path.resolve())

    if use_wandb:
        import wandb
        wandb.log({f"ik_verify/trial{trial}": wandb.Image(str(out_path))})
