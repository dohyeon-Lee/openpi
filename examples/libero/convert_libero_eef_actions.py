"""
Convert LIBERO RLDS dataset to LeRobot format with EEF state as action labels.

Action space: [ee_states(t+1)[:6], gripper_action(t)] (7D)
- ee_states(t+1): next step's EEF pose (pos 3 + ori 3) from state[:6]
- gripper_action: original gripper command from action[-1]
- Last step: ee_states duplicated (no next step)
- Noop filtering: skip steps where original action[:6] is near zero

Usage:
    python examples/libero/convert_libero_eef_actions.py --data_dir /scratch/mdorazi/libero_rlds
"""

import shutil

import numpy as np
import tensorflow_datasets as tfds
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "mdorazi/libero_spatial_eef"
RAW_DATASET_NAMES = ["libero_spatial_no_noops"]

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
                # next step EEF pose (or duplicate last)
                next_step = steps[i + 1] if i + 1 < len(steps) else steps[i]
                ee_next = next_step["observation"]["state"][:6]
                gripper_cmd = step["action"][-1:]  # (1,)
                new_action = np.concatenate([ee_next, gripper_cmd]).astype(np.float32)

                dataset.add_frame({
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": new_action,
                    "task": step["language_instruction"].decode(),
                })
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
