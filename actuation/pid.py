import torch

from state_objects.base_state_object import BaseStateObject
from utilities.tensor_utils import zeros


class PID(BaseStateObject):
    def __init__(self,
                 k_p=6.0,
                 k_i=0.01,
                 k_d=0.5,
                 min_length=100,
                 RANGE=100,
                 tol=0.15):
        """
        Initialize the PID controller.

        Args:
            k_p: Proportional gain coefficient (default: 6.0)
            k_i: Integral gain coefficient (default: 0.01)
            k_d: Derivative gain coefficient (default: 0.5)
            min_length: Minimum cable length in original units (default: 100)
            RANGE: Operating range of cable length in original units (default: 100)
            tol: Error tolerance for considering target reached (default: 0.15)
        """
        super().__init__('pid')
        # self.last_control = np.zeros(n_motor)
        self.last_error = None
        self.cum_error = None
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.min_length = min_length / 100.
        self.RANGE = RANGE / 100.
        self.tol = tol
        self.LEFT_RANGE = None
        self.RIGHT_RANGE = None
        self.done = None

    def to(self, device):
        super().to(device)
        if self.last_error is not None:
            self.last_error = self.last_error.to(device)

        if self.cum_error is not None:
            self.cum_error = self.cum_error.to(device)

        return self

    def update_control_by_target_gait(self, current_length, target_gait, rest_length):
        """Compute PID control signal to track target gait position.

        This method calculates the control input needed to move cables from their
        current length to the target gait position. It uses PID control with error
        tracking and includes logic to detect when targets are reached.

        Args:
            current_length: Current cable lengths tensor
            target_gait: Target normalized positions (0-1 range)
            rest_length: Natural/rest length of cables

        Returns:
            tuple: (u, position) where:
                - u: Control signal tensor, clipped to [-1, 1]
                - position: Normalized current position (0-1 range)

        Note:
            - Control is set to 0 for cables marked as 'done' (within tolerance)
            - Prevents slack by setting control to 0 when cable is shorter than
              rest length and control would further shorten it
        """
        if self.done is None:
            self.done = torch.tensor([False] * current_length.shape[0],
                                     device=current_length.device)

        if self.cum_error is None:
            self.cum_error = zeros(current_length.shape,
                                   ref_tensor=current_length)

        u = zeros(current_length.shape,
                  ref_tensor=current_length)

        min_length = self.min_length
        range_ = max(self.RANGE, 1e-5)

        position = (current_length - min_length) / range_

        if self.done:
            return u, position

        target_length = min_length + range_ * target_gait
        error = position - target_gait

        low_error_cond1 = torch.abs(error).flatten() < self.tol
        low_error_cond2 = torch.abs(current_length - target_length).flatten() < 0.1
        low_error_cond3 = torch.logical_and(target_gait.flatten() == 0, position.flatten() < 0)

        low_error = torch.logical_or(
            torch.logical_or(self.done, low_error_cond1),
            torch.logical_or(low_error_cond2, low_error_cond3)
        )

        self.done[low_error] = True

        d_error = zeros(error.shape, ref_tensor=error) \
            if self.last_error is None else error - self.last_error
        self.cum_error += error
        self.last_error = error

        u[~low_error] = (self.k_p * error[~low_error]
                         + self.k_i * self.cum_error[~low_error]
                         + self.k_d * d_error[~low_error])

        u = torch.clip(u, min=-1, max=1)
        slack = torch.logical_and(current_length < rest_length, u < 0)
        u[slack] = 0

        return u, position

    def reset(self):
        """Reset controller state to initial conditions.

        Clears error history, cumulative error, and done flags. Should be called
        between episodes or when starting a new control sequence.
        """
        self.last_error = None
        self.cum_error = None
        self.done = None
