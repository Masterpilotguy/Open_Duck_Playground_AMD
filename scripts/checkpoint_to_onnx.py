#!/usr/bin/env python3
"""
Standalone utility to convert any saved Orbax checkpoint into an ONNX model.

Usage:
    python scripts/checkpoint_to_onnx.py checkpoints/2026_11_01_075841_1474560
    python scripts/checkpoint_to_onnx.py checkpoints/2026_11_01_075841_1474560 --output my_model.onnx
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Suppress noisy TensorFlow initialization output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
from playground.common.export_onnx import export_onnx


def convert_checkpoint(
    checkpoint_dir: str,
    output_path: str = None,
    action_size: int = 14,
    obs_size: int = 101,
    config_name: str = "BerkeleyHumanoidJoystickFlatTerrain",
) -> str:
    ckpt_path = Path(checkpoint_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    if output_path is None:
        output_path = str(ckpt_path) + ".onnx"

    print(f"Loading checkpoint: {ckpt_path}")
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(str(ckpt_path))

    ppo_params = locomotion_params.brax_ppo_config(config_name)

    print(f"Exporting to ONNX:  {output_path}")
    export_onnx(
        params=params,
        act_size=action_size,
        ppo_params=ppo_params,
        obs_size=obs_size,
        output_path=output_path,
    )
    print(f"Successfully generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a saved raw Orbax checkpoint into an ONNX model."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the Orbax checkpoint directory to convert",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination ONNX file path (default: <checkpoint_dir>.onnx)",
    )
    parser.add_argument(
        "--action_size",
        type=int,
        default=14,
        help="Number of actuators (default: 14 for Open Duck Mini)",
    )
    parser.add_argument(
        "--obs_size",
        type=int,
        default=101,
        help="Observation vector size (default: 101)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="BerkeleyHumanoidJoystickFlatTerrain",
        help="Locomotion config preset (default: BerkeleyHumanoidJoystickFlatTerrain)",
    )

    args = parser.parse_args()
    convert_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        action_size=args.action_size,
        obs_size=args.obs_size,
        config_name=args.config,
    )


if __name__ == "__main__":
    main()

