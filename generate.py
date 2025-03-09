# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch
import random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

# Example prompts for different tasks
EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}

def _validate_args(args):
    """Validate the input arguments."""
    # Basic checks
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupported task: {args.task}"

    # Set default sampling steps based on task type
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    # Set default sample shift based on task type and size
    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # Set default frame number based on task type
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # Validate frame number for text-to-image tasks
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupported frame_num {args.frame_num} for task {args.task}"

    # Set base seed if not provided
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    # Validate size for the task
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupported size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from an image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default