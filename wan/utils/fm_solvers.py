# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
# Convert dpm solver for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import inspect
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (KarrasDiffusionSchedulers,
                                                   SchedulerMixin,
                                                   SchedulerOutput)
from diffusers.utils import deprecate, is_scipy_available
from diffusers.utils.torch_utils import randn_tensor

if is_scipy_available():
    pass


def get_sampling_sigmas(sampling_steps, shift):
    """
    Generate sigmas for sampling steps with a given shift factor.
    
    Args:
        sampling_steps (int): Number of sampling steps.
        shift (float): Shift factor to adjust the sigmas.
    
    Returns:
        np.ndarray: Array of sigmas for each sampling step.
    """
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    """
    Retrieve timesteps for the scheduler based on the provided parameters.
    
    Args:
        scheduler: The scheduler instance.
        num_inference_steps (int, optional): Number of inference steps.
        device (str or torch.device, optional): Device to move timesteps to.
        timesteps (List[int], optional): Custom timesteps.
        sigmas (List[float], optional): Custom sigmas.
        **kwargs: Additional arguments.
    
    Returns:
        Tuple: Timesteps and number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `FlowDPMSolverMultistepScheduler` is a fast dedicated high-order solver for diffusion ODEs.
    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.
    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model. This determines the resolution of the diffusion process.
        solver_order (`int`, defaults to 2):
            The DPMSolver order which can be `1`, `2`, or `3`. It is recommended to use `s