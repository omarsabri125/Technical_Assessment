import gc
import os
import platform
from typing import Any

import psutil
import torch


def gib(num_bytes: int | float) -> float:
    """Convert bytes to GiB."""
    return round(float(num_bytes) / (1024 ** 3), 4)


def process_rss_bytes() -> int:
    """Return this Python process's current resident RAM."""
    return psutil.Process(os.getpid()).memory_info().rss


def cuda_sync() -> None:
    """Wait until queued CUDA work is finished before timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clean_memory() -> None:
    """Release Python references and unused cached CUDA blocks."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def select_compute_dtype() -> torch.dtype:
    """Use BF16 when supported; otherwise use FP16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def dtype_name(dtype: torch.dtype) -> str:
    """Return a readable PyTorch dtype name."""
    return str(dtype).replace("torch.", "").upper()


def get_input_device(model: torch.nn.Module) -> torch.device:
    """Find the device containing the token embedding layer."""
    return model.get_input_embeddings().weight.device


def hardware_info() -> dict[str, Any]:
    """Collect hardware and runtime information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to collect GPU information.")

    props = torch.cuda.get_device_properties(0)

    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": gib(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
