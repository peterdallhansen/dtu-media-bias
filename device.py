"""Centralized device detection logic for all models."""

import torch


def get_device(device_setting="auto"):
    """Get the compute device based on setting and availability.

    Args:
        device_setting: Device preference - "cuda", "mps", "cpu", or "auto".
                        Each config file can specify its own DEVICE setting.

    Returns:
        torch.device: The selected compute device.
    """
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_setting == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Warning: CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    if device_setting == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("Warning: MPS requested but not available, falling back to CPU")
        return torch.device("cpu")

    return torch.device("cpu")
