from typing import Optional, Union

import torch
from flatten_dict import flatten, unflatten
from torch.nn.modules.module import T

from utilities.misc_utils import DEFAULT_DTYPE


class BaseStateObject(torch.nn.Module):
    name: str
    dtype: torch.dtype
    device: torch.device | str

    def __init__(self,
                 name: str,
                 dtype=DEFAULT_DTYPE,
                 device='cpu'):
        super().__init__()
        self.name = name
        self.dtype = dtype
        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

        return self

    def cuda(self: T, device: str | torch.device | None = None) -> T:
        super().cuda()
        self.to('cuda')

        return self

    def cpu(self: T) -> T:
        super().cpu()
        self.to('cpu')

        return self