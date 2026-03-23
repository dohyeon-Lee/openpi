import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def dls_qdot(Jv, v0, damping=1e-6):
    """
    Damped least-squares IK:
        qdot0 = J^T (J J^T + lambda I)^(-1) v0
    """
    A = Jv @ Jv.T + damping * np.eye(3)
    return Jv.T @ np.linalg.solve(A, v0)


def tangent_frame_directions(v0):
    """
    Build 6 task-space directions:
        +t, -t, +n1, -n1, +n2, -n2
    """
    norm_v = np.linalg.norm(v0)
    if norm_v < 1e-12:
        raise ValueError("v0 is too small to define tangent direction.")

    t = v0 / norm_v

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(t, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    n1 = ref - np.dot(ref, t) * t
    n1 /= (np.linalg.norm(n1) + 1e-12)

    n2 = np.cross(t, n1)
    n2 /= (np.linalg.norm(n2) + 1e-12)

    return [t, -t, n1, -n1, n2, -n2], {"t": t, "n1": n1, "n2": n2}


def box_support(weight, lower, upper):
    """
    Closed-form support of a box:
        max_{x in [lower, upper]} weight^T x
    """
    weight = np.asarray(weight)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.sum(np.where(weight >= 0.0, weight * upper, weight * lower)))


def fit_quadratic_from_three_samples(y0, y1, y2):
    """
    Fit scalar quadratic:
        y(alpha) = a0 + a1 * alpha + a2 * alpha^2

    using samples at alpha = 0, 1, 2.

    Since:
        y(0) = a0
        y(1) = a0 + a1 + a2
        y(2) = a0 + 2a1 + 4a2

    solution:
        a0 = y0
        a2 = (y2 - 2*y1 + y0) / 2
        a1 = y1 - y0 - a2
    """
    a0 = y0
    a2 = 0.5 * (y2 - 2.0 * y1 + y0)
    a1 = y1 - y0 - a2
    return a0, a1, a2


def fit_quadratic_vector_from_three_samples(y0, y1, y2):
    """
    Vector version of fit_quadratic_from_three_samples.
    """
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    a0 = y0
    a2 = 0.5 * (y2 - 2.0 * y1 + y0)
    a1 = y1 - y0 - a2
    return a0, a1, a2


# ============================================================
# Root / interval utilities
# ============================================================

def solve_quadratic_inequality_geq_zero(a2, a1, a0, eps=1e-12):
    """
    Solve:
        a2 * x^2 + a1 * x + a0 >= 0

    Returns a list of intervals:
        [(lo1, hi1), (lo2, hi2), ...]

    where bounds can be +/- np.inf.

    This is general-purpose and then we intersect with x >= 1 later.
    """
    # Degenerate to linear
    if abs(a2) < eps:
        if abs(a1) < eps:
            # Constant
            if a0 >= 0.0:
                return [(-np.inf, np.inf)]
            else:
                return []

        # Linear: a1 x + a0 >= 0
        x_star = -a0 / a1
        if a1 > 0:
            return [(x_star, np.inf)]
        else:
            return [(-np.inf, x_star)]

    # Proper quadratic
    disc = a1 * a1 - 4.0 * a2 * a0

    if disc < -eps:
        # No real roots; sign determined by a2
        if a2 > 0:
            return [(-np.inf, np.inf)]
        else:
            return []

    # Treat tiny negative as zero
    disc = max(disc, 0.0)
    sqrt_disc = np.sqrt(disc)

    r1 = (-a1 - sqrt_disc) / (2.0 * a2)
    r2 = (-a1 + sqrt_disc) / (2.0 * a2)
    lo, hi = min(r1, r2), max(r1, r2)

    if a2 > 0:
        # outside roots
        return [(-np.inf, lo), (hi, np.inf)]
    else:
        # between roots
        return [(lo, hi)]


def intersect_intervals(intervals_a, intervals_b):
    """
    Intersect two interval lists.
    """
    out = []
    for a_lo, a_hi in intervals_a:
        for b_lo, b_hi in intervals_b:
            lo = max(a_lo, b_lo)
            hi = min(a_hi, b_hi)
            if lo <= hi:
                out.append((lo, hi))
    return out


def intersect_many_interval_lists(list_of_interval_lists):
    """
    Intersect multiple interval lists.
    """
    if len(list_of_interval_lists) == 0:
        return []

    current = list_of_interval_lists[0]
    for nxt in list_of_interval_lists[1:]:
        current = intersect_intervals(current, nxt)
        if len(current) == 0:
            return []
    return current


def best_alpha_from_intervals(intervals, alpha_min=1.0, alpha_max=np.inf):
    """
    Restrict intervals to [alpha_min, alpha_max], then return the largest feasible alpha.
    """
    clipped = intersect_intervals(intervals, [(alpha_min, alpha_max)])
    if len(clipped) == 0:
        return None, clipped

    best = max(hi for _, hi in clipped)
    return best, clipped


# ============================================================
# Quadratic surrogate
# ============================================================

class QuadraticAlphaSurrogate:
    """
    Closed-form alpha* estimator with quadratic alpha dependence.

    Model
    -----
    For tar embodiment, for each task-space direction u:

        h_tar(u; alpha)
          = u^T p
            + u^T (alpha v0 dt)
            + 0.5 * u^T g(alpha) * dt^2
            + support_box( P_pos^T u ; tau_tar_box )

    where
        P_pos = 0.5 * Jv Minv dt^2

    and we approximate the dynamic term
        g(alpha) := Jdot_qdot(alpha) - Jv Minv h(alpha)
    by a quadratic fit:
        g(alpha) ~= g0 + g1 alpha + g2 alpha^2

    Then:
        h_tar(u; alpha) = c0_u + c1_u alpha + c2_u alpha^2

    img support is fixed at alpha = 1.
    """

    def __init__(
        self,
        Jv,
        M,
        p,
        v0,
        qdot0,
        h_samples,
        Jvdotqdot_samples,
        img_limits,
        tar_limits,
        dt,
        directions=None,
        eps=1e-12,
    ):
        """
        Parameters
        ----------
        Jv : (3, n)
        M  : (n, n)
        p  : (3,)
        v0 : (3,)
        qdot0 : (n,)

        h_samples:
            dict like {0.0: h0, 1.0: h1, 2.0: h2}
            where hk = h(q, k * qdot0)

        Jvdotqdot_samples:
            dict like {0.0: j0, 1.0: j1, 2.0: j2}
            where jk = Jdot(q, k*qdot0) @ (k*qdot0), translational part only

        img_limits / tar_limits:
            dict with keys
                dq_min, dq_max, tau_min, tau_max
        """
        self.Jv = np.asarray(Jv)
        self.M = np.asarray(M)
        self.Minv = np.linalg.inv(M)
        self.p = np.asarray(p)
        self.v0 = np.asarray(v0)
        self.qdot0 = np.asarray(qdot0)

        self.img_limits = img_limits
        self.tar_limits = tar_limits
        self.dt = float(dt)
        self.eps = eps

        if directions is None:
            directions, frame = tangent_frame_directions(v0)
            self.frame = frame
        else:
            self.frame = None
        self.directions = directions

        # ----------------------------------------------------
        # Torque -> position affine map coefficient
        # ----------------------------------------------------
        self.P_pos = 0.5 * self.Jv @ self.Minv * self.dt**2

        # ----------------------------------------------------
        # Build quadratic model for:
        #   g(alpha) = Jdot_qdot(alpha) - Jv Minv h(alpha)
        # ----------------------------------------------------
        h0 = np.asarray(h_samples[0.0])
        h1 = np.asarray(h_samples[1.0])
        h2 = np.asarray(h_samples[2.0])

        j0 = np.asarray(Jvdotqdot_samples[0.0])
        j1 = np.asarray(Jvdotqdot_samples[1.0])
        j2 = np.asarray(Jvdotqdot_samples[2.0])

        g0 = j0 - self.Jv @ (self.Minv @ h0)
        g1s = j1 - self.Jv @ (self.Minv @ h1)
        g2s = j2 - self.Jv @ (self.Minv @ h2)

        self.g0, self.g1, self.g2 = fit_quadratic_vector_from_three_samples(g0, g1s, g2s)

        # ----------------------------------------------------
        # Precompute per-direction coefficients
        #
        # h_tar(u; alpha) = c0_u + c1_u alpha + c2_u alpha^2
        # ----------------------------------------------------
        self.tar_coeffs = []
        self.img_supports = []

        for u in self.directions:
            u = np.asarray(u)

            # torque support for img / tar
            w = self.P_pos.T @ u
            s_img_box = box_support(w, self.img_limits["tau_min"], self.img_limits["tau_max"])
            s_tar_box = box_support(w, self.tar_limits["tau_min"], self.tar_limits["tau_max"])

            # tar support coefficients
            # p term + torque-box term + dynamic quadratic term + alpha*v0dt term
            c0_u = float(u @ self.p) + s_tar_box + 0.5 * float(u @ self.g0) * self.dt**2
            c1_u = float(u @ self.v0) * self.dt + 0.5 * float(u @ self.g1) * self.dt**2
            c2_u = 0.5 * float(u @ self.g2) * self.dt**2

            self.tar_coeffs.append((c0_u, c1_u, c2_u))

            # img support at alpha = 1 using img torque box
            g_img_at_1 = self.g0 + self.g1 + self.g2
            h_img_u = (
                float(u @ self.p)
                + float(u @ self.v0) * self.dt
                + 0.5 * float(u @ g_img_at_1) * self.dt**2
                + s_img_box
            )
            self.img_supports.append(h_img_u)

    # --------------------------------------------------------
    # Separate joint-speed upper bound
    # --------------------------------------------------------
    def alpha_vel_bound(self):
        """
        Necessary joint-velocity bound:
            |alpha * qdot0_i| <= dq_max_abs_i
        """
        dq_min = np.asarray(self.tar_limits["dq_min"])
        dq_max = np.asarray(self.tar_limits["dq_max"])

        # conservative symmetric magnitude
        dq_abs_max = np.maximum(np.abs(dq_min), np.abs(dq_max))

        ratios = []
        for i, qd in enumerate(self.qdot0):
            if abs(qd) < 1e-12:
                continue
            ratios.append(dq_abs_max[i] / abs(qd))

        if len(ratios) == 0:
            return np.inf
        return float(np.min(ratios))

    # --------------------------------------------------------
    # Direction-wise feasible alpha intervals
    # --------------------------------------------------------
    def feasible_intervals_from_directions(self, verbose=False):
        """
        For each direction u, require:
            h_img(u) <= h_tar(u; alpha)

        i.e.
            c2 alpha^2 + c1 alpha + (c0 - h_img) >= 0

        This yields a feasible interval list for each direction.
        Intersect them all.
        """
        interval_lists = []

        for idx, ((c0, c1, c2), h_img) in enumerate(zip(self.tar_coeffs, self.img_supports)):
            a2 = c2
            a1 = c1
            a0 = c0 - h_img

            intervals = solve_quadratic_inequality_geq_zero(a2, a1, a0, eps=self.eps)

            if verbose:
                print(f"[dir {idx}] inequality: {a2:.6e} a^2 + {a1:.6e} a + {a0:.6e} >= 0")
                print(f"[dir {idx}] intervals = {intervals}")

            if len(intervals) == 0:
                return [], False

            interval_lists.append(intervals)

        feasible = intersect_many_interval_lists(interval_lists)
        return feasible, len(feasible) > 0

    # --------------------------------------------------------
    # Final alpha*
    # --------------------------------------------------------
    def compute_alpha_star(self, verbose=False):
        """
        Final result:
            alpha* = largest alpha in
                     (intersection of all direction-feasible intervals)
                     intersect [1, alpha_vel_bound]

        Returns a dict.
        """
        alpha_vel = self.alpha_vel_bound()
        intervals, ok = self.feasible_intervals_from_directions(verbose=verbose)

        if not ok:
            return {
                "feasible": False,
                "alpha_star": 1.0,
                "reason": "no feasible alpha interval from direction constraints",
                "alpha_vel_bound": alpha_vel,
                "intervals": [],
            }

        alpha_star, clipped = best_alpha_from_intervals(
            intervals,
            alpha_min=1.0,
            alpha_max=alpha_vel,
        )

        if alpha_star is None:
            return {
                "feasible": False,
                "alpha_star": 1.0,
                "reason": "direction-feasible intervals do not overlap with [1, alpha_vel_bound]",
                "alpha_vel_bound": alpha_vel,
                "intervals": intervals,
            }

        return {
            "feasible": True,
            "alpha_star": float(alpha_star),
            "reason": "ok",
            "alpha_vel_bound": float(alpha_vel),
            "intervals": clipped,
        }
        
        
################################################################################

import numpy as np
import time

# ------------------------------------------------------------
# Paste / import the first code block before running this block
# ------------------------------------------------------------

def bias_model_demo(alpha, qdot0):
    """
    Toy bias model for demonstration only.

    In real use, replace with:
        h(alpha) = bias_terms(q, alpha * qdot0)

    Here we make a simple model:
        h(alpha) = g + alpha * d + alpha^2 * c
    """
    g = np.array([0.8, -0.4, 0.3, 0.2, -0.1, 0.05, -0.02])
    d = np.array([0.10, 0.06, -0.04, 0.03, 0.02, -0.01, 0.01])
    c = np.array([0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02])
    return g + alpha * d + (alpha ** 2) * c


def jdot_qdot_demo(alpha, qdot0):
    """
    Toy translational Jdot*qdot model for demonstration only.

    In real use, replace with:
        Jvdotqdot(alpha) = jdot_qdot_linear(q, alpha * qdot0)

    Here:
        j(alpha) = j0 + alpha * j1 + alpha^2 * j2
    """
    j0 = np.array([0.000, 0.000, 0.000])
    j1 = np.array([0.010, -0.004, 0.003])
    j2 = np.array([0.020, -0.006, 0.005])
    return j0 + alpha * j1 + (alpha ** 2) * j2


def demo():
    # --------------------------------------------------------
    # Example 7-DoF same morphology
    # --------------------------------------------------------
    n = 7
    dt = 0.03

    # Toy translational Jacobian
    Jv = np.array([
        [0.25,  0.05, 0.00, 0.10, 0.00, 0.03, 0.00],
        [0.00,  0.20, 0.04, 0.00, 0.08, 0.00, 0.02],
        [0.03,  0.00, 0.18, 0.00, 0.00, 0.06, 0.05],
    ])

    # Toy SPD mass matrix
    M = np.diag([3.0, 2.5, 2.2, 1.8, 1.5, 1.2, 1.0])

    # Current EE position and original chunk velocity
    p = np.array([0.45, -0.10, 0.32])
    v0 = np.array([0.22, 0.00, 0.00])

    # Nominal joint velocity from DLS
    qdot0 = dls_qdot(Jv, v0, damping=1e-6)

    # Build samples at alpha = 0, 1, 2
    h_samples = {
        0.0: bias_model_demo(0.0, qdot0),
        1.0: bias_model_demo(1.0, qdot0),
        2.0: bias_model_demo(2.0, qdot0),
    }

    Jvdotqdot_samples = {
        0.0: jdot_qdot_demo(0.0, qdot0),
        1.0: jdot_qdot_demo(1.0, qdot0),
        2.0: jdot_qdot_demo(2.0, qdot0),
    }

    img_limits = {
        "dq_min": -np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5]),
        "dq_max":  np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5]),
        "tau_min": -np.array([15.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0]),
        "tau_max":  np.array([15.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0]),
    }

    tar_limits = {
        "dq_min": -np.array([1.6, 1.6, 1.6, 1.8, 1.8, 2.0, 2.0]),
        "dq_max":  np.array([1.6, 1.6, 1.6, 1.8, 1.8, 2.0, 2.0]),
        "tau_min": -np.array([25.0, 25.0, 20.0, 16.0, 12.0, 10.0, 8.0]),
        "tau_max":  np.array([25.0, 25.0, 20.0, 16.0, 12.0, 10.0, 8.0]),
    }

    directions, frame = tangent_frame_directions(v0)

    t0 = time.perf_counter()

    surrogate = QuadraticAlphaSurrogate(
        Jv=Jv,
        M=M,
        p=p,
        v0=v0,
        qdot0=qdot0,
        h_samples=h_samples,
        Jvdotqdot_samples=Jvdotqdot_samples,
        img_limits=img_limits,
        tar_limits=tar_limits,
        dt=dt,
        directions=directions,
    )

    result = surrogate.compute_alpha_star(verbose=True)

    t1 = time.perf_counter()

    print("\n================ RESULT ================")
    print(f"feasible        : {result['feasible']}")
    print(f"reason          : {result['reason']}")
    print(f"alpha_star      : {result['alpha_star']:.6f}")
    print(f"alpha_vel_bound : {result['alpha_vel_bound']:.6f}")
    print(f"intervals       : {result['intervals']}")
    print(f"elapsed [ms]    : {(t1 - t0) * 1000.0:.3f}")

    print("\nqdot0 =", qdot0)
    print("tangent =", frame['t'])
    print("normal1 =", frame['n1'])
    print("normal2 =", frame['n2'])


if __name__ == "__main__":
    demo()