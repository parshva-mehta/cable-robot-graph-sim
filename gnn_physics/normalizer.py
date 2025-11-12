from typing import Tuple

import torch

from state_objects.base_state_object import BaseStateObject
from utilities.misc_utils import DEFAULT_DTYPE


class AccumulatedNormalizer(BaseStateObject):

    def __init__(self,
                 shape: Tuple,
                 max_acc_steps: int = 2000,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 name: str = 'unknown'):
        """
        Normalizer that accumulates during first epoch to compute mean and std of features

        @param shape: shape of feature
        @param max_acc_steps: max number of accumulation steps
        @param dtype: data type for torch tensors
        @param name: feature name
        """
        super().__init__('normalizer')
        self.shape = shape
        zeros = torch.zeros(shape, dtype=dtype, device=self.device)

        self.accum_flag = False
        self.name = name

        self._max_acc_steps = max_acc_steps
        self._num_accumulations = torch.tensor(0, dtype=torch.int)
        self._acc_count = torch.tensor(0, dtype=torch.int)
        self._acc_sum = zeros.clone()
        self._acc_sum_squared = zeros.clone()

        self._std_epsilon = zeros + 1e-3

    def start_accum(self):
        self.accum_flag = True

    def stop_accum(self):
        self.accum_flag = False

    def __call__(self, batched_data: torch.Tensor, no_accum: bool = False) -> torch.Tensor:
        """
        normalizer function
        @param batched_data:
        @return:
        """
        if self.accum_flag and not no_accum:
            data_acc_sum = batched_data.detach().sum(dim=0, keepdim=True)
            data_acc_sum_squared = (batched_data.detach() ** 2).sum(dim=0, keepdim=True)
            num_data = batched_data.shape[0]

            if len(batched_data.shape) == 3:
                data_acc_sum = data_acc_sum.sum(dim=2)
                data_acc_sum_squared = data_acc_sum_squared.sum(dim=2)
                num_data *= batched_data.shape[2]

            self._num_accumulations = self._num_accumulations + 1
            self._acc_count = self._acc_count + num_data
            self._acc_sum = self._acc_sum + data_acc_sum
            self._acc_sum_squared = self._acc_sum_squared + data_acc_sum_squared

        mean, std = self.mean, self.std_w_eps
        if len(batched_data.shape) == 3:
            mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)

        normalized = (batched_data - mean) / std
        return normalized

    def accum(self, batched_data):
        data_acc_sum = batched_data.detach().sum(dim=0, keepdim=True)
        data_acc_sum_squared = (batched_data.detach() ** 2).sum(dim=0, keepdim=True)
        num_data = batched_data.shape[0]

        if len(batched_data.shape) == 3:
            data_acc_sum = data_acc_sum.sum(dim=2)
            data_acc_sum_squared = data_acc_sum_squared.sum(dim=2)
            num_data *= batched_data.shape[2]

        self._num_accumulations = self._num_accumulations + 1
        self._acc_count = self._acc_count + num_data
        self._acc_sum += data_acc_sum
        self._acc_sum_squared += data_acc_sum_squared

    def to(self, device):
        super().to(device)
        self._acc_count = self._acc_count.to(device)
        self._num_accumulations = self._num_accumulations.to(device)
        self._acc_sum = self._acc_sum.to(device)
        self._acc_sum_squared = self._acc_sum_squared.to(device)
        self._std_epsilon = self._std_epsilon.to(device)

        return self

    def inverse(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        """Inverse transformation of the normalizer."""
        mean, std = self.mean, self.std_w_eps
        if len(normalized_batch_data.shape) == 3:
            mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)

        return normalized_batch_data * std + mean

    def inverse_no_mean(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        return normalized_batch_data * self.std_w_eps

    @property
    def _safe_count(self):
        # To ensure count is at least one and avoid nan's.
        return max(self._acc_count, 1)

    @property
    def mean(self):
        return self._acc_sum / self._safe_count

    @property
    def std(self):
        var = self._acc_sum_squared / self._safe_count - self.mean ** 2
        var = torch.clamp_min(var, 0.)
        return torch.sqrt(var)

    @property
    def std_w_eps(self):
        # To use in case the std is too small.
        return torch.maximum(self.std, self._std_epsilon)


class DummyNormalizer(AccumulatedNormalizer):
    def __init__(self,
                 shape: Tuple,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 name: str = 'unknown'):
        super().__init__(shape,
                         0,
                         dtype,
                         name)
        self._mean = torch.zeros(shape, dtype=dtype, device=self.device)
        self._std = torch.ones(shape, dtype=dtype, device=self.device)

    def to(self, device):
        super().to(device)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        return self

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def std_w_eps(self):
        return self._std
