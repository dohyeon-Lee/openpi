"""Inspect LIBERO RLDS dataset structure."""

import tensorflow_datasets as tfds

DATA_DIR = "/scratch/mdorazi/libero_rlds"
DATASET_NAME = "libero_spatial_no_noops"

ds = tfds.load(DATASET_NAME, data_dir=DATA_DIR, split="train")
episode = next(iter(ds))
steps = list(episode["steps"].as_numpy_iterator())
step = steps[0]

print("=== observation keys ===")
for k, v in step["observation"].items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print(f"\n=== action ===")
print(f"  shape={step['action'].shape}")
print(f"  values={step['action']}")

print(f"\n=== language instruction ===")
print(f"  {step['language_instruction'].decode()}")
print("state:", step["observation"]["state"])
print("joint_state:", step["observation"]["joint_state"])
# print(step)