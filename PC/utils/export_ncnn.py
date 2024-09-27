#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export NanoDet model to NCNN format using PNNX.

This script automates the process of converting a NanoDet PyTorch model to NCNN format.
It exports the model to TorchScript, checks for the PNNX tool (ensuring it's in PATH),
and uses PNNX to convert the model to NCNN format.

Usage:
    python export_ncnn.py --cfg_path config.yml --model_path model.pth --input_shape 320,320

Author: Your Name
Date: YYYY-MM-DD
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


def check_pnnx(prefix="PNNX:"):
    """
    Check if PNNX is available in the system PATH.

    Args:
        prefix (str): Prefix for logging messages.

    Returns:
        Path: Path object pointing to the PNNX executable.

    Raises:
        SystemExit: If PNNX is not found in the system PATH.
    """
    from shutil import which

    pnnx_executable = "pnnx.exe" if os.name == "nt" else "pnnx"
    pnnx_path = which(pnnx_executable)

    if pnnx_path:
        print(f"{prefix} PNNX is available at: {pnnx_path}")
        return Path(pnnx_path)
    else:
        print(
            f"{prefix} PNNX not found in system PATH. Please install PNNX or ensure it's in your PATH."
        )
        sys.exit(1)


def export_torchscript(config, model_path, output_path, input_shape):
    """
    Export NanoDet model to TorchScript.

    Args:
        config (dict): Configuration dictionary.
        model_path (str): Path to the trained model weights.
        output_path (Path): Path to save the TorchScript model.
        input_shape (tuple): Input shape as (height, width).

    Returns:
        Path: Path to the saved TorchScript model.
    """
    logger = Logger(local_rank=-1, save_dir=config.save_dir, use_tensorboard=False)

    # Create model and load weights
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)

    # Convert backbone weights for RepVGG models
    if config.model.arch.backbone.name == "RepVGG":
        deploy_config = config.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)

    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TorchScript: tracing the model with dummy inputs
    with torch.no_grad():
        dummy_input = torch.zeros(
            1, 3, input_shape[0], input_shape[1]
        )  # Batch size = 1
        model.eval().cpu()
        model_traced = torch.jit.trace(model, example_inputs=dummy_input).eval()
        model_traced.save(output_path)
        print(f"Finished export to TorchScript: {output_path}")

    return output_path


def convert_to_ncnn(torchscript_path, input_shape, output_dir, device="cpu", fp16=False):
    """
    Convert TorchScript model to NCNN format using PNNX.

    Args:
        torchscript_path (Path): Path to the TorchScript model.
        input_shape (tuple): Input shape as (height, width).
        output_dir (Path): Directory to save the NCNN model files.
        device (str): Device to use for export ('cpu' or 'cuda').
        fp16 (bool): Whether to enable FP16 precision.

    Returns:
        tuple: Paths to the NCNN param and bin files.
    """
    prefix = "NCNN:"
    pnnx_path = check_pnnx(prefix=prefix)
    pnnx = str(pnnx_path.resolve())

    # Prepare arguments for PNNX
    ncnn_param = output_dir / "model.ncnn.param"
    ncnn_bin = output_dir / "model.ncnn.bin"
    ncnn_py = output_dir / "model_ncnn.py"

    cmd = [
        pnnx,
        str(torchscript_path),
        f"inputshape=[1,3,{input_shape[0]},{input_shape[1]}]",
        f"device={device}",
        f"ncnnparam={ncnn_param}",
        f"ncnnbin={ncnn_bin}",
        f"ncnnpy={ncnn_py}",
    ]
    if fp16:
        cmd.append("fp16=1")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{prefix} Running PNNX command:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{prefix} PNNX conversion failed: {e}")
        sys.exit(1)

    print(f"{prefix} NCNN model files generated at: {output_dir}")
    return ncnn_param, ncnn_bin


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert NanoDet model to NCNN format.",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Path to the NanoDet configuration YAML file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained NanoDet model weights (.pth or .ckpt).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="nanodet_ncnn_model",
        help="Directory to save NCNN model files.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="Model input shape as 'height,width'. Example: 320,320",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for model export ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision for NCNN model.",
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    out_dir = Path(args.out_dir)
    input_shape = args.input_shape
    device = args.device
    fp16 = args.fp16

    # Load configuration
    load_config(cfg, cfg_path)

    # Determine input shape
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        try:
            input_shape = tuple(map(int, input_shape.split(",")))
            assert len(input_shape) == 2, "Input shape must be 'height,width'."
        except Exception as e:
            print(f"Error parsing input_shape: {e}")
            sys.exit(1)

    # Determine model path
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best", "model_best.ckpt")
        if not Path(model_path).is_file():
            print(
                f"Default model path {model_path} does not exist. Please specify --model_path."
            )
            sys.exit(1)

    # Convert model path to absolute path
    model_path = Path(model_path).resolve()

    # Step 1: Export to TorchScript
    torchscript_path = out_dir / "nanodet.torchscript.pt"
    export_torchscript(cfg, model_path, torchscript_path, input_shape)

    # Step 2: Convert to NCNN using PNNX
    convert_to_ncnn(
        torchscript_path,
        input_shape,
        out_dir,
        device=device,
        fp16=fp16,
    )

    print(f"NCNN model files are saved in: {out_dir}")


if __name__ == "__main__":
    main()
