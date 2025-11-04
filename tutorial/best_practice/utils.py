import numpy as np
import torch
import triton.runtime.driver as driver


def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def round_up(x: torch.Tensor, value: int):
    assert isinstance(value, int)
    assert x.dtype == torch.int32
    return torch.div(x + (value - 1), value, rounding_mode='trunc') *value


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
