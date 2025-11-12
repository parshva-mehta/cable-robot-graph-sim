from collections import OrderedDict
from typing import Dict, Union, Tuple, List

import torch

from robots.cable_driven_robot import CableDrivenRobot
from state_objects.cables import get_cable, ActuatedCable, Cable
from state_objects.system_topology import SystemTopology
from state_objects.tensegrity_rods import TensegrityRod, TensegrityHousingRod
from state_objects.primitive_shapes import Cylinder
from utilities import torch_quaternion


class TensegrityRobot(CableDrivenRobot):
    system_topology: SystemTopology
    rods: Dict[str, TensegrityRod]
    cables: Dict[str, Cable]
    num_rods: int
    pos: torch.Tensor
    linear_vel: torch.Tensor
    quat: torch.Tensor
    ang_vel: torch.Tensor
    actuated_cables: Dict[str, ActuatedCable]
    non_actuated_cables: Dict[str, Cable]
    k_mat: torch.Tensor
    c_mat: torch.Tensor
    cable2rod_idxs: Dict
    rod_end_pts: torch.Tensor
    cable_map: Dict

    def __init__(self, cfg: Dict):
        """
        Tensegrity robot class

        @param cfg: config dict
        """
        rods = self._init_rods(cfg)
        cables = self._init_cables(cfg)

        super().__init__(rods, cables)

    @property
    def sphere_radius(self):
        return list(self.rods.values())[0].sphere_radius

    @property
    def rod_length(self):
        return list(self.rigid_bodies.values())[0].length

    def _init_rods(self, config: dict) -> Dict[str, TensegrityRod]:
        """
        Instantiate rod objects
        @param config: config containing rod configs
        @return: dictionary of rod name to rod object
        """
        rods = OrderedDict()
        for rod_config in config['rods']:
            # rod_state = TensegrityRod.init_from_cfg(rod_config)
            rod_state = TensegrityHousingRod.init_from_cfg(rod_config)
            rods[rod_state.name] = rod_state

        return rods

    def _init_cables(self, config: dict) -> Dict[str, Cable]:
        """
        Instantiate cable objects
        @param config: config containing cable configs
        @return: dictionary of cable name to cable object
        """
        cables = OrderedDict()
        for cable_config in config['cables']:
            cable_cls = get_cable(cable_config['type'])
            config = {k: v for k, v in cable_config.items() if k != 'type'}

            cable = cable_cls.init_from_cfg(
                config
            )
            cables[cable.name] = cable

        return cables

    def compute_end_pts(self, state):
        end_pts = []
        length = self.rod_length
        for i, rod in enumerate(self.rods.values()):
            state_ = state[:, i * 13: i * 13 + 7]
            principal_axis = torch_quaternion.compute_prin_axis(state_[:, 3:7])
            end_pts.extend(Cylinder.compute_end_pts_from_state(state_, principal_axis, length))

        return torch.concat(end_pts, dim=2)


class TensegrityRobotGNN(TensegrityRobot):
    _inv_mass: torch.Tensor | None
    _inv_inertia: torch.Tensor | None
    _cable_damping: torch.Tensor | None
    _cable_stiffness: torch.Tensor | None
    _cable_rest_length: torch.Tensor | None
    _cable_template: torch.Tensor | None
    _contact_nodes: List | None
    node_mapping: Dict
    template: List
    template_idx: torch.Tensor
    num_nodes: int
    rod_body_verts: List
    body_verts: torch.Tensor
    num_nodes_per_rod: int
    num_bodies: int


    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._inv_mass = None
        self._inv_inertia = None

        self._cable_damping = None
        self._cable_stiffness = None
        self._cable_rest_length = None


        self.node_mapping = {
            body.name: i * len(rod.rigid_bodies) + j
            for i, rod in enumerate(self.rods.values())
            for j, body in enumerate(rod.rigid_bodies.values())
        }

        self._cable_template = None
        self.template = self.get_template_graph()
        self.template_idx = self.convert_to_idx_edges(self.template)

        self.num_nodes = len(self.node_mapping)
        self.rod_body_verts = [rod.body_verts for rod in self.rods.values()]
        self.body_verts = torch.vstack(self.rod_body_verts)
        self.num_nodes_per_rod = self.body_verts.shape[0] // len(self.rods)
        self.num_bodies = max(self.node_mapping.values()) + 1

        self._contact_nodes = None

    def to(self, device):
        super().to(device)
        if self._cable_damping is not None:
            self._cable_damping = self._cable_damping.to(device)
        if self._cable_stiffness is not None:
            self._cable_stiffness = self._cable_stiffness.to(device)
        if self._cable_rest_length is not None:
            self._cable_rest_length = self._cable_rest_length.to(device)
        if self._inv_mass is not None:
            self._inv_mass = self._inv_mass.to(device)
        if self._inv_inertia is not None:
            self._inv_inertia = self._inv_inertia.to(device)

        return self

    def update_state(self, next_state, update_sys_top=False):
        batch_size = next_state.shape[0]
        next_state_ = next_state.reshape(-1, 13, 1)

        self.pos = next_state_[:, :3].reshape(batch_size, -1, 1)
        self.quat = next_state_[:, 3:7].reshape(batch_size, -1, 1)
        self.linear_vel = next_state_[:, 7:10].reshape(batch_size, -1, 1)
        self.ang_vel = next_state_[:, 10:].reshape(batch_size, -1, 1)

        # Update each rod
        for i, rod in enumerate(self.rods.values()):
            rod.update_state(
                self.pos[:, i * 3: (i + 1) * 3],
                self.linear_vel[:, i * 3: (i + 1) * 3],
                self.quat[:, i * 4: (i + 1) * 4],
                self.ang_vel[:, i * 3: (i + 1) * 3],
            )

        if update_sys_top:
            self.update_system_topology()

    def convert_to_idx_edges(self, edges):
        return torch.tensor([
            [self.node_mapping[s0], self.node_mapping[s1]]
            for s0, s1 in edges
        ], dtype=torch.int).T

    def get_contact_nodes(self):
        if self._contact_nodes is None:
            self._contact_nodes = [
                v for k, v in self.node_mapping.items()
                if 'sphere' in k
            ]

        return self._contact_nodes

    def get_template_graph(self):
        template = [
            edge
            for rod in self.rods.values()
            for edge in rod.get_template_graph()
        ]

        return template

    def get_cable_edge_idxs(self):
        if self._cable_template is None:
            mapping = {}
            i, j = 0, 0
            for k in self.node_mapping.keys():
                if "sphere" in k:
                    mapping[str(i)] = self.node_mapping[k]
                    i += 1
                if "motor" in k:
                    mapping["b" + str(j)] = self.node_mapping[k]
                    j += 1

            endpt_fn = lambda x, idx: mapping[x.split("_")[idx]]
            if len(list(self.cables.values())[0].end_pts[0].split("_")) > 2:
                self._cable_template = torch.tensor([
                    [endpt_fn(e, 1), endpt_fn(e, 2)]
                    for cable in self.cables.values()
                    for e in cable.end_pts
                ], dtype=torch.int).T
            else:
                self._cable_template = torch.tensor([
                    [endpt_fn(cable.end_pts[0], 1), endpt_fn(cable.end_pts[1], 1)] if i % 2 == 0 else
                    [endpt_fn(cable.end_pts[1], 1), endpt_fn(cable.end_pts[0], 1)]
                    for cable in self.cables.values()
                    for i in range(2)
                ], dtype=torch.int).T

        return self._cable_template

    @property
    def end_pts(self):
        end_pts = []
        for rod in self.rods.values():
            end_pts.extend(rod.end_pts)

        return end_pts

    @property
    def cable_damping(self):
        if self._cable_damping is None:
            self._cable_damping = torch.vstack([
                s.damping
                for cable in self.cables.values()
                for s in [cable, cable]
            ]).reshape(-1, 1)
        return self._cable_damping

    @property
    def cable_stiffness(self):
        if self._cable_stiffness is None:
            self._cable_stiffness = torch.vstack([
                s.stiffness
                for cable in self.cables.values()
                for s in [cable, cable]
            ]).reshape(-1, 1)
        return self._cable_stiffness

    @property
    def cable_rest_length(self):
        if self._cable_rest_length is None:
            self._cable_rest_length = torch.vstack([
                s._rest_length
                for cable in self.cables.values()
                for s in [cable, cable]
            ])
        return self._cable_rest_length

    @property
    def inv_mass(self):
        if self._inv_mass is None:
            self._inv_mass = torch.vstack([
                1 / n.mass
                for r in self.rods.values()
                for n in r.rigid_bodies.values()
            ]).reshape(-1, 1)
        return self._inv_mass

    @property
    def inv_inertia(self):
        if self._inv_inertia is None:
            self._inv_inertia = torch.vstack([
                torch.diagonal(n.I_body_inv)
                for r in self.rods.values()
                for n in r.rigid_bodies.values()
            ])
        return self._inv_inertia

