"""Inspect a LIBERO HDF5 file to see its structure and available keys."""

import h5py
import numpy as np


def print_structure(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{indent}{name}/")


HDF5_PATH = "/scratch/mdorazi/libero_data/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"

with h5py.File(HDF5_PATH, "r") as f:
    print("=== HDF5 Structure ===")
    f.visititems(print_structure)

    print("\n=== First demo, first step sample values ===")
    demo_key = list(f["data"].keys())[0]
    demo = f["data"][demo_key]
    print(f"demo: {demo_key}, num steps: {demo['actions'].shape[0]}")
    print(f"action shape: {demo['actions'].shape}")

    
    if "obs" in demo:
        for k in demo["obs"].keys():
            v = demo["obs"][k]
            print(f"obs/{k}: shape={v.shape}, first={v[0] if v.ndim == 1 else v.shape}")

 
