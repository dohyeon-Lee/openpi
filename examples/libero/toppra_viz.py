"""
TOPP-RA helpers for trajectory smoothing and visualisation.

JointToppraPlanner  — joint-space TOPP-RA (proper, with real joint limits)
ToppraVisualizer    — legacy Cartesian TOPP-RA (kept for reference)

Usage in debug_joints.py:
    from toppra_viz import JointToppraPlanner

    planner = JointToppraPlanner(controller_freq=ctrl_freq)
    eef_smooth = planner.plan(q_waypoints, fk_pos=kin.fk_pos)
    # collect: toppra_predictions.append((step, eef_smooth))
    # then:    planner.add_to_plot(axes, toppra_predictions, colors)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Franka Panda joint limits
PANDA_VEL_LIMITS    = np.array([2.175, 2.175, 2.175, 2.175, 2.61,  2.61,  2.61 ])  # rad/s
PANDA_ACC_LIMITS    = np.array([15.0,  7.5,   10.0,  12.5,  15.0,  20.0,  20.0 ])  # rad/s²
PANDA_TORQUE_LIMITS = np.array([87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0 ])  # N·m
PANDA_FRICTION_COEF = np.array([0.2,   0.2,   0.15,  0.15,  0.1,   0.1,   0.1  ])  # viscous friction (N·m·s/rad)


class JointToppraPlanner:
    """
    Joint-space TOPP-RA planner.

    Pipeline:
        q_waypoints (T, N)  — from diff_ik_trajectory
        → SplineInterpolator (joint space)
        → TOPP-RA with joint vel/acc [+ torque] constraints
        → sample at controller_freq → q_smooth (M, N)
        → FK → eef_smooth (M, 3)  [for visualisation]

    토크 제약 사용법:
        planner = JointToppraPlanner(controller_freq=ctrl_freq)
        eef = planner.plan(q_waypoints, fk_pos=kin.fk_pos, inv_dyn=kin.inv_dyn)
    """

    def __init__(
        self,
        controller_freq: float = 20.0,
        vel_limits:    np.ndarray = PANDA_VEL_LIMITS,
        acc_limits:    np.ndarray = PANDA_ACC_LIMITS,
        torque_limits: np.ndarray = PANDA_TORQUE_LIMITS,
        friction_coef: np.ndarray = PANDA_FRICTION_COEF,
    ):
        self.controller_freq = controller_freq
        self.vel_limits    = np.asarray(vel_limits,    dtype=float)
        self.acc_limits    = np.asarray(acc_limits,    dtype=float)
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        self.friction_coef = np.asarray(friction_coef, dtype=float)
        self._available = self._check_toppra()

    def _check_toppra(self) -> bool:
        try:
            import toppra  # noqa: F401
            return True
        except ImportError:
            logger.warning("toppra not installed – TOPP-RA disabled")
            return False

    def plan(self, q_waypoints: np.ndarray, fk_pos, inv_dyn=None, qdot_end_override: np.ndarray = None, qdot_start_override: np.ndarray = None):
        """
        Run joint-space TOPP-RA and return smoothed EEF positions.

        Parameters
        ----------
        q_waypoints : (T, N)  joint waypoints from diff_ik_trajectory
        fk_pos      : callable (N,) → (3,)         forward kinematics
        inv_dyn     : callable (q, qd, qdd) → tau  inverse dynamics (optional).
                      제공 시 토크 제약이 추가됨. e.g. kin.inv_dyn

        Returns
        -------
        eef_smooth : (M, 3) EEF positions sampled at controller_freq,
                     or None if toppra unavailable / computation fails.
        """
        if not self._available:
            return None, None

        from toppra.interpolator import SplineInterpolator
        from toppra.algorithm import TOPPRA
        from toppra.constraint import (
            JointVelocityConstraint,
            JointAccelerationConstraint,
            JointTorqueConstraint,
        )

        q_waypoints = np.asarray(q_waypoints, dtype=float)   # (T, N)
        T, N = q_waypoints.shape

        # ── timestamps: real time in seconds (ctrl_freq 기준) ──────────────────
        ts = np.linspace(0, (T - 1) / self.controller_freq, T, dtype=float)
        dt = ts[1] - ts[0]

        # ── spline: bc_type으로 시작/끝 joint velocity를 경계조건으로 직접 지정 ──
        # (참고: RACE_code/toppra_interpolator.py 방식)
        qdot_start = (q_waypoints[1]  - q_waypoints[0])  / dt  # (N,) rad/s
        qdot_end   = (q_waypoints[-1] - q_waypoints[-2]) / dt  # (N,) rad/s
        if qdot_start_override is not None:
            qdot_start = np.asarray(qdot_start_override, dtype=float)
        if qdot_end_override is not None:
            qdot_end = np.asarray(qdot_end_override, dtype=float)
        try:
            path = SplineInterpolator(
                ts, q_waypoints,
                bc_type=((1, qdot_start), (1, qdot_end)),
            )
        except Exception as exc:
            logger.warning("TOPP-RA spline failed: %s", exc)
            return None

        # ── constraints ────────────────────────────────────────────────────────
        vlim = np.column_stack([-self.vel_limits[:N],    self.vel_limits[:N]])    # (N, 2)
        alim = np.column_stack([-self.acc_limits[:N],    self.acc_limits[:N]])    # (N, 2)
        tlim = np.column_stack([-self.torque_limits[:N], self.torque_limits[:N]]) # (N, 2)

        constraints = [
            JointVelocityConstraint(vlim),
            JointAccelerationConstraint(alim),
        ]
        if inv_dyn is not None:
            constraints.append(
                JointTorqueConstraint(inv_dyn, tlim, fs_coef=self.friction_coef[:N])
            )

        # ── sd_start/sd_end: joint velocity → path parameter velocity ─────────
        def _qdot_to_sd(qdot: np.ndarray, s: float) -> float:
            dqds = path(np.array([s]), 1)[0]   # path tangent at s
            denom = np.dot(dqds, dqds)
            if denom < 1e-12:
                return 0.0
            return float(np.clip(np.dot(qdot, dqds) / denom, 0.0, None))

        sd_start = _qdot_to_sd(qdot_start, ts[0])
        sd_end   = _qdot_to_sd(qdot_end,   ts[-1])

        try:
            instance = TOPPRA(constraints, path, parametrizer="ParametrizeConstAccel")
            traj = instance.compute_trajectory(sd_start, sd_end)
        except Exception as exc:
            logger.warning("TOPP-RA failed: %s", exc)
            return None, None

        if traj is None:
            logger.warning("TOPP-RA returned None trajectory")
            return None, None

        # ── sample at controller_freq ──────────────────────────────────────────
        num_ts = max(int(np.ceil(traj.duration * self.controller_freq)), 1)
        t_sample = np.linspace(0, traj.duration, num_ts)
        q_smooth = traj(t_sample, 0)   # (M, N)

        # ── FK → EEF positions ─────────────────────────────────────────────────
        try:
            eef_smooth = np.array([fk_pos(q) for q in q_smooth])   # (M, 3)
        except Exception as exc:
            logger.warning("FK failed during TOPP-RA back-conversion: %s", exc)
            return None, None

        return eef_smooth, q_smooth   # (M, 3), (M, N)

    def add_to_plot(self, axes, toppra_predictions, colors=("orange", "limegreen", "deepskyblue")):
        """Overlay smoothed EEF trajectories onto existing x/y/z axes."""
        for i, (start_step, eef_smooth) in enumerate(toppra_predictions):
            if eef_smooth is None:
                continue
            t_range = list(range(start_step, start_step + len(eef_smooth)))
            for dim, (ax, color) in enumerate(zip(axes, colors)):
                ax.plot(
                    t_range, eef_smooth[:, dim],
                    color=color, alpha=0.6, linewidth=1.5,
                    linestyle=":",
                    label="accel_chunk" if i == 0 else None,
                )


# ── Legacy: Cartesian TOPP-RA (kept for reference) ────────────────────────────

class ToppraVisualizer:
    """Cartesian-space TOPP-RA (deprecated — use JointToppraPlanner instead)."""

    def __init__(self, controller_freq=20.0, policy_freq=4.0, vel_limit=0.3, acc_limit=1.0):
        self.controller_freq = controller_freq
        self.policy_freq     = policy_freq
        self.vel_limit       = vel_limit
        self.acc_limit       = acc_limit
        self._available      = self._check_toppra()

    def _check_toppra(self):
        try:
            import toppra  # noqa: F401
            return True
        except ImportError:
            logger.warning("toppra not installed")
            return False

    def apply(self, predicted_eef, start_eef, start_vel=None):
        if not self._available:
            return None
        from toppra.interpolator import SplineInterpolator
        from toppra.algorithm import TOPPRA
        from toppra.constraint import JointVelocityConstraint, JointAccelerationConstraint

        ndim      = 3
        waypoints = np.asarray(predicted_eef, dtype=float)
        start     = np.asarray(start_eef,     dtype=float)
        sv        = np.zeros(ndim) if start_vel is None else np.asarray(start_vel, dtype=float)
        N         = len(waypoints)

        try:
            path = SplineInterpolator(
                np.linspace(0, N / self.policy_freq, N + 1, endpoint=True),
                np.vstack([start, waypoints]),
                bc_type=((1, sv), "natural"),
            )
        except Exception as exc:
            logger.warning("TOPP-RA spline failed: %s", exc)
            return None

        vlim = np.tile([-self.vel_limit, self.vel_limit], (ndim, 1))
        alim = np.tile([-self.acc_limit, self.acc_limit], (ndim, 1))
        instance = TOPPRA(
            [JointVelocityConstraint(vlim), JointAccelerationConstraint(alim)],
            path,
            parametrizer="ParametrizeConstAccel",
        )
        try:
            traj = instance.compute_trajectory(0, 0)
        except Exception as exc:
            logger.warning("TOPP-RA solver failed: %s", exc)
            return None
        if traj is None:
            return None

        num_ts = max(int(np.ceil(traj.duration * self.controller_freq)), 1)
        ts     = np.linspace(0, num_ts / self.controller_freq, num_ts)
        return traj(ts, 0)

    def add_to_plot(self, axes, toppra_predictions, colors=("r", "g", "b")):
        for i, (start_step, qs) in enumerate(toppra_predictions):
            if qs is None:
                continue
            t_range = list(range(start_step, start_step + len(qs)))
            for dim, (ax, color) in enumerate(zip(axes, colors)):
                ax.plot(t_range, qs[:, dim], color=color, alpha=0.6,
                        linewidth=1.5, linestyle=":",
                        label="accel_chunk" if i == 0 else None)
