from typing import Dict, Union

import torch

from state_objects.base_state_object import BaseStateObject
from state_objects.cables import ActuatedCable, Cable
from state_objects.composite_body import CompositeBody
from state_objects.system_topology import SystemTopology


class CableDrivenRobot(BaseStateObject):
    rigid_bodies: Dict[str, CompositeBody]
    cables: Dict[str, Cable]
    num_bodies: int
    pos: torch.Tensor
    linear_vel: torch.Tensor
    quat: torch.Tensor
    ang_vel: torch.Tensor
    actuated_cables: Dict[str, ActuatedCable]
    non_actuated_cables: Dict[str, Cable]

    def __init__(self,
                 rigid_bodies: Dict[str, CompositeBody],
                 cables: Dict[str, Cable]
                 ):
        """
        Tensegrity robot class

        @param cfg: config dict
        """
        super().__init__('cable_driven_robot')
        self.rigid_bodies = rigid_bodies
        self.cables = cables

        self.system_topology = SystemTopology(
            list(rigid_bodies.values()),
            list(cables.values())
        )

        self.num_bodies = len(self.rigid_bodies)

        # Concat of state vars
        self.pos = torch.hstack([body.pos for body in self.rigid_bodies.values()])
        self.linear_vel = torch.hstack([body.linear_vel for body in self.rigid_bodies.values()])
        self.quat = torch.hstack([body.quat for body in self.rigid_bodies.values()])
        self.ang_vel = torch.hstack([body.ang_vel for body in self.rigid_bodies.values()])

        # Split cables to actuated and non-actuated
        self.actuated_cables, self.non_actuated_cables = {}, {}
        for k, cable in self.cables.items():
            if isinstance(cable, ActuatedCable):
                self.actuated_cables[k] = cable
            else:
                self.non_actuated_cables[k] = cable

    def to(self, device: Union[str, torch.device]):
        for k, body in self.rigid_bodies.items():
            body.to(device)

        for k, cable in self.cables.items():
            cable.to(device)

        return self

    def update_state(self, next_state):
        batch_size = next_state.shape[0]
        next_state_ = next_state.reshape(-1, 13, 1)

        self.pos = next_state_[:, :3].reshape(batch_size, -1, 1)
        self.quat = next_state_[:, 3:7].reshape(batch_size, -1, 1)
        self.linear_vel = next_state_[:, 7:10].reshape(batch_size, -1, 1)
        self.ang_vel = next_state_[:, 10:].reshape(batch_size, -1, 1)

        # Update each body
        for i, body in enumerate(self.rigid_bodies.values()):
            body.update_state(
                self.pos[:, i * 3: (i + 1) * 3],
                self.linear_vel[:, i * 3: (i + 1) * 3],
                self.quat[:, i * 4: (i + 1) * 4],
                self.ang_vel[:, i * 3: (i + 1) * 3],
            )

    def get_curr_state(self):
        state = torch.hstack([
            self.pos.reshape(-1, 3, 1),
            self.quat.reshape(-1, 4, 1),
            self.linear_vel.reshape(-1, 3, 1),
            self.ang_vel.reshape(-1, 3, 1),
        ]).reshape(-1, 13 * len(self.rigid_bodies), 1)

        return state
