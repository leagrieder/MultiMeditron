from typing import Union
import torch

def get_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)

    return dtype
