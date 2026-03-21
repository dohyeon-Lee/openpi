"""
Inspect raw RLDS data: compare EEF pos vs state-action vs delta for a few episodes.

Columns per episode:
  col 0 (Absolute) : eef_pos, state_action (next eef)
  col 1 (Delta)    : Δeef, orig_action (OSC), Δstate_action, Δstate_action/0.05 (scaled)

Usage:
    python examples/libero/inspect_stateaction.py --data_dir /data2/dohyeon/libero_data
    python examples/libero/inspect_stateaction.py --data_dir /data2/dohyeon/libero_data --n_episodes 3
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


OSC_POS_SCALE = 0.05   # osc_pose.json output_max for xyz
OSC_ORI_SCALE = 0.5    # osc_pose.json output_max for rotation


def plot_episode(ax_rows, steps, ep_idx):
    """
    ax_rows: list of 3 axes (one per xyz) × 2 columns = shape (3, 2)
      col 0: absolute values  (eef_pos, state_action)
      col 1: delta values     (Δeef, orig_action, Δstate_action, Δstate_action/0.05)
    """
    eef_pos = np.array([s["observation"]["state"][:3] for s in steps])   # (T, 3)
    orig_act = np.array([s["action"][:3]              for s in steps])   # (T, 3) OSC delta [-1,1]
    state_act = np.array([
        (steps[i + 1] if i + 1 < len(steps) else steps[i])["observation"]["state"][:3]
        for i in range(len(steps))
    ])  # (T, 3) next eef pos

    delta_eef    = np.diff(eef_pos, axis=0, prepend=eef_pos[:1])  # (T, 3) meters
    delta_state  = state_act - eef_pos                             # (T, 3) meters
    delta_scaled = delta_state / OSC_POS_SCALE                    # (T, 3) ~[-1, 1]

    T = len(steps)
    t = np.arange(T)
    labels_xyz = ["x", "y", "z"]

    for dim in range(3):
        # --- absolute ---
        ax = ax_rows[dim][0]
        ax.plot(t, eef_pos[:, dim],   label="eef_pos (curr)",        color="blue",   lw=1.5)
        ax.plot(t, state_act[:, dim], label="state_action (next eef)", color="orange", lw=1.5, ls="--")
        ax.set_ylabel(f"{labels_xyz[dim]} (abs, m)")
        if dim == 0:
            ax.set_title(f"Ep{ep_idx} — Absolute")
        if dim == 2:
            ax.set_xlabel("timestep")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # --- delta ---
        ax = ax_rows[dim][1]
        ax.plot(t, delta_eef[:, dim],    label="Δeef (m)",               color="blue",       lw=1.5)
        ax.plot(t, orig_act[:, dim],     label="orig OSC action [-1,1]",  color="red",        lw=1.5, ls="--")
        ax.plot(t, delta_state[:, dim],  label="Δstate (m)",              color="orange",     lw=1.2, ls=":")
        ax.plot(t, delta_scaled[:, dim], label="Δstate/0.05 (scaled)",    color="green",      lw=1.5, ls="-.")
        ax.set_ylabel(f"{labels_xyz[dim]} (delta)")
        if dim == 0:
            ax.set_title(f"Ep{ep_idx} — Delta  (green≈red → scaling OK)")
        if dim == 2:
            ax.set_xlabel("timestep")
        ax.legend(fontsize=7, loc="upper right")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset_name", default="libero_spatial_no_noops")
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--out", default="inspect_stateaction.png")
    args = parser.parse_args()

    raw_dataset = tfds.load(args.dataset_name, data_dir=args.data_dir, split="train")

    fig, axes = plt.subplots(
        nrows=3, ncols=2 * args.n_episodes,
        figsize=(7 * args.n_episodes, 9),
    )
    # axes shape: (3, 2*n_episodes)
    # episode k → columns [2k, 2k+1]

    for ep_idx, episode in enumerate(raw_dataset.take(args.n_episodes)):
        steps = list(episode["steps"].as_numpy_iterator())
        col_offset = ep_idx * 2
        ax_rows = [[axes[dim][col_offset], axes[dim][col_offset + 1]] for dim in range(3)]
        plot_episode(ax_rows, steps, ep_idx)
        print(f"Episode {ep_idx}: {len(steps)} steps")

    fig.suptitle(
        "Absolute: eef_pos vs state_action(next eef)\n"
        "Delta: Δeef(m) | orig OSC[-1,1] | Δstate(m) | Δstate/0.05(scaled)  ←  green≈red이면 스케일링 OK",
        fontsize=9,
    )
    fig.tight_layout()
    out = pathlib.Path(args.out)
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
