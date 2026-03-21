"""
Convert LIBERO RLDS dataset to LeRobot format with state-based delta actions.

Action space: [delta_eef_pos/0.05 (3), delta_eef_ori/0.5 (3), gripper (1)]
- delta_eef = next_eef_state - curr_eef_state  (actual robot movement, dynamics-aware)
- Divided by OSC output scale (pos: 0.05m, ori: 0.5rad) → maps to ~[-1, 1], compatible with env.step()
- gripper: original commanded gripper value

Motivation: raw OSC teleop commands are aggressive (human intent), whereas actual EEF
delta reflects what the robot physically realized (smoother, dynamics-filtered).

Usage:
uv run examples/libero/convert_libero_stateaction_to_lerobot.py --data_dir /data2/dohyeon/libero_data
"""

import shutil

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "dohyeon/libero_stateaction"
RAW_DATASET_NAMES = [
    "libero_spatial_no_noops",
]

# OSC output scale (osc_pose.json: output_max)
OSC_POS_SCALE = 0.05   # meters
OSC_ORI_SCALE = 0.5    # radians


def main(data_dir: str, *, push_to_hub: bool = False):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            steps = list(episode["steps"].as_numpy_iterator())
            for i, step in enumerate(steps):
                next_step = steps[i + 1] if i + 1 < len(steps) else step
                curr_eef = step["observation"]["state"][:6]         # current EEF pos+ori
                next_eef = next_step["observation"]["state"][:6]    # next EEF pos+ori
                delta_eef = next_eef - curr_eef                     # actual robot movement
                # Scale to ~[-1, 1] to match OSC input convention
                delta_eef[:3] /= OSC_POS_SCALE
                delta_eef[3:6] /= OSC_ORI_SCALE
                gripper = step["action"][6:7]                       # commanded gripper
                state_action = np.concatenate([delta_eef, gripper])

                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": state_action,
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
