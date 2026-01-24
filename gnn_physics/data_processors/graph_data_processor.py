"""Graph data processor for tensegrity robot GNN physics simulation.

This module handles the conversion of robot states into graph-structured data
suitable for GNN-based physics prediction. It manages node and edge features,
normalization, and batching for efficient training and inference.
"""
from enum import Enum
from typing import List, Tuple, Union, NamedTuple, Dict

import torch
from torch_geometric.data import Data as GraphData

from gnn_physics.normalizer import AccumulatedNormalizer, DummyNormalizer
from robots.tensegrity import TensegrityRobotGNN
from state_objects.base_state_object import BaseStateObject
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE
from utilities.tensor_utils import zeros, safe_norm


class NodeFeats(NamedTuple):
    """Node feature container for graph nodes.

    Stores all features associated with graph nodes including velocities,
    mass properties, spatial relationships, and simulation metadata.
    """
    node_vel: torch.Tensor
    node_inv_mass: torch.Tensor
    node_inv_inertia: torch.Tensor
    node_dir_from_com: torch.Tensor
    node_dist_from_com_norm: torch.Tensor
    node_dist_to_ground: torch.Tensor
    node_body_verts: torch.Tensor
    node_dist_to_first_node: torch.Tensor
    node_dist_to_first_node_norm: torch.Tensor
    node_prin_axis: torch.Tensor
    node_pos: torch.Tensor
    node_prev_pos: torch.Tensor
    node_sim_type: torch.Tensor
    body_mask: torch.Tensor


class BodyEdgeFeats(NamedTuple):
    """Edge features for rigid body connections.

    Contains distance and rest length information for edges connecting
    nodes within the same rigid body.
    """
    body_dist: torch.Tensor
    body_dist_norm: torch.Tensor
    body_rest_dist: torch.Tensor
    body_rest_dist_norm: torch.Tensor


class CableEdgeFeats(NamedTuple):
    """Edge features for cable connections.

    Stores cable-specific features including distances, velocities, stiffness,
    damping, control signals, and actuation status.
    """
    cable_dist: torch.Tensor
    cable_dist_norm: torch.Tensor
    cable_dir: torch.Tensor
    cable_rel_vel_norm: torch.Tensor
    cable_rest_length: torch.Tensor
    cable_stiffness: torch.Tensor
    cable_damping: torch.Tensor
    cable_ctrls: torch.Tensor
    cable_actuated_mask: torch.Tensor


class ContactEdgeFeats(NamedTuple):
    """Edge features for ground contact interactions.

    Contains contact geometry, velocities, and proximity information for
    edges between robot nodes and ground.
    """
    contact_dist: torch.Tensor
    contact_normal: torch.Tensor
    contact_tangent: torch.Tensor
    contact_rel_vel_normal: torch.Tensor
    contact_rel_vel_tangent: torch.Tensor
    contact_close_mask: torch.Tensor


class CacheableFeats(NamedTuple):
    """Container for features that can be precomputed and cached.

    Stores batch-size-specific features that remain constant during simulation
    and can be cached for performance optimization.
    """
    node_inv_mass: torch.Tensor
    node_inv_inertia: torch.Tensor
    node_body_verts: torch.Tensor
    body_rest_dist: torch.Tensor
    body_rest_dist_norm: torch.Tensor
    cable_stiffness: torch.Tensor
    cable_damping: torch.Tensor
    cable_actuated_mask: torch.Tensor
    contact_normal: torch.Tensor
    body_edge_idx: torch.Tensor
    cable_edge_idx: torch.Tensor
    contact_edge_idx: torch.Tensor
    body_mask: torch.Tensor


class GraphFeats(NamedTuple):
    """
    Complete graph feature set ready for GNN processing.

    Contains all normalized node features, edge features, edge indices,
    needed for graph neural network forward pass.
    """
    node_x: torch.Tensor
    body_edge_attr: torch.Tensor
    body_edge_idx: torch.Tensor
    cable_edge_attr: torch.Tensor
    cable_edge_idx: torch.Tensor
    contact_edge_attr: torch.Tensor
    contact_edge_idx: torch.Tensor
    contact_close_mask: torch.Tensor
    node_hidden_state: torch.Tensor


class PredGnnAttrs(NamedTuple):
    """GNN prediction outputs and intermediate values.

    Stores predicted positions, velocities, and various delta-velocity
    representations used during training and evaluation.
    """
    pos: torch.Tensor
    vel: torch.Tensor
    p_pos: torch.Tensor
    p_vel: torch.Tensor
    pf_dv: torch.Tensor
    p_dv: torch.Tensor
    norm_dv: torch.Tensor
    body_mask: torch.Tensor
    node_hidden_state: torch.Tensor


class CableInputType(Enum):
    """Enum defining how cable actuation is represented in the graph.

    REST_LENS: Cable actuation encoded as target rest lengths
    CTRLS: Cable actuation encoded as control signals
    """
    REST_LENS = 'rest_lens'
    CTRLS = 'ctrls'


class GraphDataProcessor(BaseStateObject):
    """Processes tensegrity robot states into graph-structured data for GNN physics prediction.

    This class handles the complex transformation from SE(3) robot states to graph representations
    suitable for graph neural networks. It manages:
    - Node feature computation (mass, inertia, velocities, spatial relationships)
    - Edge feature computation (body, cable, and contact edges)
    - Feature normalization for stable training
    - Batch processing and caching for performance
    - Conversion between pose representations and node positions
    """
    robot: TensegrityRobotGNN
    MAX_DIST_TO_GRND: float
    CONTACT_EDGE_THRESHOLD: float
    NUM_OUT_STEPS: int
    NUM_CTRLS_HIST: int
    NUM_DATASETS: int
    node_hidden_state_size: int
    cable_input_type: CableInputType
    node_feat_dict: Dict
    body_edge_feat_dict: Dict
    cable_edge_feat_dict: Dict
    contact_edge_feat_dict: Dict
    hier_node_feat_dict: Dict
    hier_edge_feat_dict: Dict
    dt: torch.Tensor
    normalizers: Dict

    def __init__(self,
                 tensegrity: TensegrityRobotGNN,
                 con_edge_threshold: float = 2e-1,
                 num_out_steps: int = 1,
                 dt: float = 0.01,
                 max_dist_to_grnd: float = 0.5,
                 cache_batch_sizes: List | None = None,
                 num_datasets: int = 10,
                 node_hidden_state_size: int = 1024,
                 num_ctrls_hist: int = 2,
                 cable_input_type: CableInputType = CableInputType.CTRLS):
        """Initialize the graph data processor.

        Args:
            tensegrity: TensegrityRobotGNN instance defining robot structure
            con_edge_threshold: Distance threshold (m) for creating ground contact edges
            num_out_steps: Number of future timesteps to predict
            dt: Simulation timestep size (seconds)
            max_dist_to_grnd: Maximum distance to ground for feature clipping (m)
            cache_batch_sizes: List of batch sizes to precompute and cache
            num_datasets: Number of different datasets for one-hot encoding
            node_hidden_state_size: Dimension of node hidden state vectors
            num_ctrls_hist: Number of historical control steps to include
            cable_input_type: How cable actuation is encoded (REST_LENS or CTRLS)
        """
        super().__init__('graph data processor')
        with torch.no_grad():
            self.MAX_DIST_TO_GRND = max_dist_to_grnd
            self.CONTACT_EDGE_THRESHOLD = con_edge_threshold
            self.NUM_OUT_STEPS = num_out_steps
            self.NUM_CTRLS_HIST = num_ctrls_hist
            self.NUM_DATASETS = num_datasets

            self.node_hidden_state_size = node_hidden_state_size
            self.cable_input_type = cable_input_type

            self.node_feat_dict = {
                'node_vel': 3,
                'node_inv_mass': 1,
                'node_inv_inertia': 3,
                'node_dist_to_ground': 1,
                'node_body_verts': 3,
                'node_dist_to_first_node': 3,
                'node_dist_to_first_node_norm': 1,
                'node_dir_from_com': 3,
                'node_dist_from_com_norm': 1,
                'node_prin_axis': 3,
                'node_sim_type': num_datasets
            }

            self.body_edge_feat_dict = {
                'body_dist': 3,
                'body_dist_norm': 1,
                'body_rest_dist': 3,
                'body_rest_dist_norm': 1,
            }

            self.cable_edge_feat_dict = {
                'cable_dist': 3,
                'cable_dist_norm': 1,
                'cable_dir': 3,
                'cable_rel_vel_norm': 1,
                'cable_stiffness': 1,
                'cable_damping': 1,
                'cable_rest_length': 1 if cable_input_type == CableInputType.CTRLS else self.NUM_OUT_STEPS
            }

            if cable_input_type == CableInputType.CTRLS:
                self.cable_edge_feat_dict['cable_ctrls'] = num_ctrls_hist + num_out_steps

            self.contact_edge_feat_dict = {
                'contact_dist': 3,
                'contact_normal': 3,
                'contact_tangent': 3,
                'contact_rel_vel_normal': 1,
                'contact_rel_vel_tangent': 1,
            }

            self.hier_node_feat_dict = {
                'node': self.node_feat_dict
            }
            self.hier_edge_feat_dict = {
                'body': self.body_edge_feat_dict,
                'cable': self.cable_edge_feat_dict,
                'contact': self.contact_edge_feat_dict
            }

            self.dt = torch.tensor([[dt]], dtype=DEFAULT_DTYPE)
            self.robot = tensegrity

            # Compute node and edge feat sizes, used for initializing encoders' input size
            self.node_feat_lens = {k: sum(v.values()) for k, v in self.hier_node_feat_dict.items()}
            self.edge_feat_lens = {k: sum(v.values()) for k, v in self.hier_edge_feat_dict.items()}

            # flatten node and edge feats dicts to initialize feat normalizers
            flatten_node_feats = {k2: v
                                  for k1, d in self.hier_node_feat_dict.items()
                                  for k2, v in d.items()}
            flatten_edge_feats = {k2: v
                                  for k1, d in self.hier_edge_feat_dict.items()
                                  for k2, v in d.items()}

            # Initialize normalizer dict
            self.normalizers = {
                k: AccumulatedNormalizer((1, v), name=k, dtype=self.dtype)
                for k, v in {**flatten_node_feats, **flatten_edge_feats}.items()
            }

            if self.cable_input_type == CableInputType.CTRLS:
                self.normalizers['cable_ctrls'] = DummyNormalizer(
                    (1, self.hier_edge_feat_dict['cable']['cable_ctrls']),
                    name='cable_ctrls',
                    dtype=self.dtype,
                )

            self.normalizers['node_sim_type'] = DummyNormalizer(
                (1, 1),
                name='node_sim_type',
                dtype=self.dtype,
            )

            self.normalizers['node_dv'] = AccumulatedNormalizer(
                (1, 3 * num_out_steps),
                name='node_dv',
                dtype=self.dtype
            )
            self.normalizers['cable_dl'] = AccumulatedNormalizer(
                (1, num_out_steps),
                name='cable_dl',
                dtype=self.dtype
            )

            robot_rods = list(self.robot.rods.values())
            self.first_node_idx = robot_rods[0].sphere_idx0
            self.last_node_idx = robot_rods[-1].sphere_idx1 + sum([r.body_verts.shape[0] for r in robot_rods[:-1]])
            self.sphere0_idx = robot_rods[0].sphere_idx0
            self.sphere1_idx = robot_rods[0].sphere_idx1
            self.sphere_radius = robot_rods[0].sphere_radius.squeeze(-1)

            contact_node_idx = self.robot.num_nodes
            self.body_edge_idx_template = self._body_edge_index()
            self.cable_edge_idx_template = self._get_cable_edge_idxs()
            self.contact_edge_idx_template = self._contact_edge_index(contact_node_idx)

            self.body_mask = self._get_body_mask(1, self.device)

            self.robot_inv_mass = torch.vstack([
                self.robot.inv_mass, torch.zeros_like(self.robot.inv_mass[:1])
            ])
            self.robot_inv_inertia = torch.vstack([
                self.robot.inv_inertia.clone(), torch.zeros_like(self.robot.inv_inertia[:1])
            ])

            self.robot_cable_stiffness = self.robot.cable_stiffness.clone()
            self.robot_cable_damping = self.robot.cable_damping.clone()

            self.body_verts = self.robot.body_verts.squeeze(-1)
            self.body_verts = torch.vstack((self.body_verts, torch.zeros_like(self.body_verts[:1])))

            body_senders_idx, body_rcvrs_idx = self.body_edge_idx_template[0], self.body_edge_idx_template[1]
            self.body_rest_dists = (
                    self.body_verts[body_rcvrs_idx] - self.body_verts[body_senders_idx]
            )
            self.body_rest_dists_norm = self.body_rest_dists.norm(dim=1, keepdim=True)

            n_rods = len(self.robot.rods) * 2
            body_rcvrs = torch.tensor(
                [[-1] * n_rods + [1] * n_rods], device=self.device,
            ).reshape(-1, 1)
            self.contact_normal = body_rcvrs * torch.tensor(
                [[0., 0., 1.]],
                dtype=self.dtype,
                device=self.device
            )

            num_act_cables = len(self.robot.actuated_cables) * 2
            num_nonact_cables = len(self.robot.non_actuated_cables) * 2
            self.cable_act_mask = torch.tensor(
                [True] * num_act_cables + [False] * num_nonact_cables,
                device=self.device,
            ).reshape(-1, 1)

            self._feats_batch_cache = {}
            if cache_batch_sizes is not None:
                self.precompute_and_cache_batch_sizes(cache_batch_sizes)

    def to(self, device: Union[str, torch.device]):
        """Move all tensors and sub-objects to specified device.

        Args:
            device: Target device (e.g., 'cpu', 'cuda', or torch.device)

        Returns:
            Self for method chaining
        """
        super().to(device)
        self.robot.to(device)
        self.dt = self.dt.to(device)
        self.sphere_radius = self.sphere_radius.to(device)
        self.body_mask = self.body_mask.to(device)

        self.body_edge_idx_template = self.body_edge_idx_template.to(device)
        self.cable_edge_idx_template = self.cable_edge_idx_template.to(device)
        self.contact_edge_idx_template = self.contact_edge_idx_template.to(device)

        self.robot_inv_mass = self.robot_inv_mass.to(device)
        self.robot_inv_inertia = self.robot_inv_inertia.to(device)
        self.robot_cable_stiffness = self.robot_cable_stiffness.to(device)
        self.robot_cable_damping = self.robot_cable_damping.to(device)

        self.body_verts = self.body_verts.to(device)
        self.body_rest_dists = self.body_rest_dists.to(device)
        self.body_rest_dists_norm = self.body_rest_dists_norm.to(device)

        self.contact_normal = self.contact_normal.to(device)

        for normalizer in self.normalizers.values():
            normalizer.to(device)

        for k, cache in self._feats_batch_cache.items():
            tmp_dict = cache._asdict()
            for kk, v in tmp_dict.items():
                tmp_dict[kk] = v.to(device)
            self._feats_batch_cache[k] = CacheableFeats(**tmp_dict)

        return self

    @property
    def cached_batch_size_keys(self):
        """Get list of batch sizes that have been precomputed and cached.

        Returns:
            List of integer batch sizes available in cache
        """
        return list(self._feats_batch_cache.keys())

    def precompute_and_cache_batch_sizes(self, batch_sizes, overwrite=False):
        """Precompute and cache features for specific batch sizes.

        This optimization avoids recomputing static features during training.

        Args:
            batch_sizes: List of batch sizes to precompute
            overwrite: If True, recompute even if already cached
        """
        for bsize in batch_sizes:
            if overwrite or bsize not in self._feats_batch_cache:
                self._feats_batch_cache[bsize] = self._batch_feats(bsize)

    def start_normalizers(self):
        """Start accumulating statistics in all feature normalizers.

        Call this at the beginning of a training data collection phase to
        gather mean and standard deviation statistics for normalization.
        """
        for normalizer in self.normalizers.values():
            normalizer.start_accum()

    def stop_normalizers(self):
        """Stop accumulating statistics in all feature normalizers.

        Call this after collecting training data to finalize normalization
        parameters before training begins.
        """
        for normalizer in self.normalizers.values():
            normalizer.stop_accum()

    def _batch_feats(self, bsize: int):
        """Precompute batch-size-specific features for caching.

        Creates batched versions of static features (mass, inertia, edge indices)
        that don't change during simulation.

        Args:
            bsize: Batch size to precompute for

        Returns:
            CacheableFeats containing all batch-size-specific features
        """
        nnodes = self.contact_edge_idx_template.max() + 1
        body_edge_idx = self.batch_edge_index(self.body_edge_idx_template, bsize, nnodes)
        cable_edge_idx = self.batch_edge_index(self.cable_edge_idx_template, bsize, nnodes)
        contact_edge_idx = self.batch_edge_index(self.contact_edge_idx_template, bsize, nnodes)

        robot_inv_mass = self.robot_inv_mass.repeat(bsize, 1)
        robot_inv_inertia = self.robot_inv_inertia.repeat(bsize, 1)
        robot_cable_stiffness = self.robot_cable_stiffness.repeat(bsize, 1)
        robot_cable_damping = self.robot_cable_damping.repeat(bsize, 1)

        body_verts = self.body_verts.repeat(bsize, 1)
        body_rest_dists = self.body_rest_dists.repeat(bsize, 1)
        body_rest_dists_norm = self.body_rest_dists_norm.repeat(bsize, 1)

        contact_normal = self.contact_normal.repeat(bsize, 1)
        body_mask = self.body_mask.repeat(bsize, 1)

        cable_act_mask = self.cable_act_mask.repeat(bsize, 1)

        return CacheableFeats(
            node_inv_mass=robot_inv_mass,
            node_inv_inertia=robot_inv_inertia,
            node_body_verts=body_verts,
            body_rest_dist=body_rest_dists,
            body_rest_dist_norm=body_rest_dists_norm,
            contact_normal=contact_normal,
            cable_stiffness=robot_cable_stiffness,
            cable_damping=robot_cable_damping,
            cable_actuated_mask=cable_act_mask,
            body_edge_idx=body_edge_idx,
            cable_edge_idx=cable_edge_idx,
            contact_edge_idx=contact_edge_idx,
            body_mask=body_mask,
        )

    def batch_edge_index(self,
                         edge_index: torch.Tensor,
                         batch_size: int,
                         num_nodes: torch.Tensor,
                         ) -> torch.Tensor:
        """Expand edge indices from single graph to batch of graphs.

        Assumes all graphs in batch have identical topology (same size and connections).

        Args:
            edge_index: [2, num_edges] edge connectivity for single graph
            batch_size: Number of graphs in batch
            num_nodes: Number of nodes per graph

        Returns:
            [2, batch_size * num_edges] batched edge indices
        """
        # Assume graphs are the same size and have the same connections
        senders = edge_index[:1].repeat(batch_size, 1)
        receivers = edge_index[1:].repeat(batch_size, 1)

        offsets = num_nodes * torch.arange(
            0, batch_size,
            dtype=torch.int,
            device=senders.device
        ).reshape(-1, 1)

        senders = (senders + offsets).reshape(1, -1)
        receivers = (receivers + offsets).reshape(1, -1)

        edge_indices = torch.vstack([senders, receivers])
        return edge_indices

    def node2pose(self,
                  node_pos: torch.Tensor,
                  prev_node_pos: torch.Tensor,
                  num_nodes: int,
                  **kwargs):
        """Convert node positions to SE(3) rigid body poses.

        Computes center of mass position, orientation quaternion, linear velocity,
        and angular velocity from node positions.

        Args:
            node_pos: [batch_size * num_nodes, 3] current node positions
            prev_node_pos: [batch_size * num_nodes, 3] previous node positions
            num_nodes: Number of nodes per rigid body

        Returns:
            [batch_size, 13 * num_rods, num_timesteps] SE(3) state tensor
            Format: [pos(3), quat(4), lin_vel(3), ang_vel(3)] per rod
        """

        def compute_state(node_pos, prev_node_pos):
            curr_com_pos = node_pos.reshape(-1, num_nodes, 3).mean(dim=1)
            prev_com_pos = prev_node_pos.reshape(-1, num_nodes, 3).mean(dim=1)

            lin_vel = (curr_com_pos - prev_com_pos).unsqueeze(-1) / self.dt

            idx_0 = self.sphere0_idx
            idx_1 = self.sphere1_idx

            curr_sphere0 = node_pos[idx_0::num_nodes]
            curr_sphere1 = node_pos[idx_1::num_nodes]
            prev_sphere0 = prev_node_pos[idx_0::num_nodes]
            prev_sphere1 = prev_node_pos[idx_1::num_nodes]

            curr_prin = safe_norm(curr_sphere1 - curr_sphere0).unsqueeze(-1)
            prev_prin = safe_norm(prev_sphere1 - prev_sphere0).unsqueeze(-1)

            ang_vel = torch_quaternion.compute_ang_vel_vecs(prev_prin, curr_prin, self.dt)
            quat = torch_quaternion.compute_quat_btwn_z_and_vec(curr_prin)

            n_rods = len(self.robot.rods)
            state = torch.hstack([curr_com_pos.unsqueeze(-1), quat, lin_vel, ang_vel])
            state = state.reshape(-1, state.shape[1] * n_rods, 1)

            return state

        node_pos = node_pos.reshape(node_pos.shape[0], node_pos.shape[1], -1)
        prev_node_pos = prev_node_pos.unsqueeze(-1)
        all_node_pos = torch.cat([prev_node_pos, node_pos], dim=-1)

        states = []
        for i in range(node_pos.shape[-1]):
            node_pos = all_node_pos[..., i + 1]
            prev_node_pos = all_node_pos[..., i]

            se3_state = compute_state(node_pos, prev_node_pos)
            states.append(se3_state)

        states = torch.cat(states, dim=-1)
        return states

    def _normalize_and_hstack(self, raw_feats, feat_dict):
        """Normalize and concatenate features according to feature dictionary.

        Args:
            raw_feats: NamedTuple containing raw feature tensors
            feat_dict: Dictionary mapping feature names to dimensions

        Returns:
            Concatenated normalized feature tensor
        """
        feats_list = [
            self.normalizers[k](getattr(raw_feats, k))
            for k in feat_dict.keys()
        ]
        feats = torch.hstack(feats_list)
        return feats

    def get_normalize_feats(
            self,
            node_raw_feats: NodeFeats,
            body_edge_feats: BodyEdgeFeats,
            cable_edge_feats: CableEdgeFeats,
            contact_edge_feats: ContactEdgeFeats
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize and concatenate all features for GNN input.

        Applies learned normalization (mean/std) to each feature type and
        concatenates them into single tensors per feature type.

        Args:
            node_raw_feats: Raw node features
            body_edge_feats: Raw body edge features
            cable_edge_feats: Raw cable edge features
            contact_edge_feats: Raw contact edge features

        Returns:
            Tuple of (node_x, body_edge_attr, cable_edge_attr, contact_edge_attr)
            All features normalized and concatenated
        """
        node_x = self._normalize_and_hstack(
            node_raw_feats, self.node_feat_dict
        )
        body_edge_attr = self._normalize_and_hstack(
            body_edge_feats, self.body_edge_feat_dict
        )
        cable_edge_attr = self._normalize_and_hstack(
            cable_edge_feats, self.cable_edge_feat_dict
        )
        contact_edge_attr = self._normalize_and_hstack(
            contact_edge_feats, self.contact_edge_feat_dict
        )

        return node_x, body_edge_attr, cable_edge_attr, contact_edge_attr

    def _inject_grnd_feat(self, feat, grnd_val_tensor):
        """Add ground node feature to batched node features.

        Appends ground node feature values to the end of each graph's node features.

        Args:
            feat: [batch_size * num_robot_nodes, feat_dim] node features
            grnd_val_tensor: [1, feat_dim] ground node feature value

        Returns:
            [batch_size * (num_robot_nodes + 1), feat_dim] features with ground
        """
        num_nodes = self.robot.num_nodes
        hsize = feat.shape[1]

        feat = feat.reshape(-1, num_nodes * hsize)
        grnd_val_tensor = grnd_val_tensor.repeat(feat.shape[0], 1)
        feat_w_grnd = torch.hstack([feat, grnd_val_tensor]).reshape(-1, hsize)

        return feat_w_grnd

    def pose2node(self,
                  pos: torch.Tensor,
                  quat: torch.Tensor,
                  batch_size: int,
                  augment_grnd=False,
                  ) -> torch.Tensor:
        """Convert SE(3) rigid body poses to 3D node positions.

        Transforms body-frame node positions to world frame using position and orientation.

        Args:
            pos: [batch_size * num_rods, 3, 1] body positions
            quat: [batch_size * num_rods, 4, 1] body orientations (quaternions)
            batch_size: Number of graphs in batch
            augment_grnd: If True, append ground node at origin

        Returns:
            [batch_size * num_nodes, 3] world-frame node positions
        """
        # Get positions of nodes in body frame
        body_verts = torch.vstack(
            [r.body_verts.transpose(0, 2) for r in self.robot.rods.values()]
        ).to(pos.device).repeat(batch_size, 1, 1)

        # Rotate and translate body verts to world frame
        node_pos = torch_quaternion.rotate_vec_quat(quat, body_verts)
        node_pos = node_pos + pos
        node_pos = node_pos.transpose(1, 2).reshape(-1, 3)

        if augment_grnd:
            grnd_node_pos = zeros((batch_size, 3), ref_tensor=node_pos)
            node_pos = torch.hstack([node_pos.reshape(batch_size, -1), grnd_node_pos])
            node_pos = node_pos.reshape(-1, 3)

        return node_pos

    def _get_body_verts(self, batch_size, device):
        """Get batched body-frame vertex positions.

        Args:
            batch_size: Number of graphs in batch
            device: Target device for tensors

        Returns:
            [batch_size * num_rods, 3, num_verts] body-frame vertices
        """
        body_verts = (self.robot.body_verts
                      .to(device)
                      .transpose(0, 2)
                      .repeat(batch_size, 1, 1))
        return body_verts

    def _compute_shape_feats(self, node_pos, batch_size):
        """Compute shape-based features relative to robot frame.

        Calculates node positions in a robot-local coordinate frame defined by
        first and last nodes. Assumes no ground node in input.

        Args:
            node_pos: [batch_size * num_nodes, 3] node positions
            batch_size: Number of graphs in batch

        Returns:
            Tuple of (dist_first_node, dist_first_node_norm):
                - dist_first_node: [num_nodes, 3] position in robot frame
                - dist_first_node_norm: [num_nodes, 1] distance magnitude
        """
        num_nodes = node_pos.shape[0] // batch_size

        first_node = node_pos[self.first_node_idx::num_nodes].repeat(1, num_nodes).reshape(-1, 3)
        last_node = node_pos[self.last_node_idx::num_nodes].repeat(1, num_nodes).reshape(-1, 3)

        x_dir = torch.hstack([
            (last_node - first_node)[:, :2],
            torch.zeros_like(last_node[:, :1])
        ])
        x_dir = safe_norm(x_dir)
        z_dir = torch.tensor(
            [[0, 0, 1]],
            dtype=self.dtype,
            device=self.device
        ).repeat(x_dir.shape[0], 1)
        y_dir = torch.cross(z_dir, x_dir, dim=1)
        y_dir = safe_norm(y_dir)
        rot_mat = torch.stack([x_dir, y_dir, z_dir], dim=2)

        dist_first_node = (node_pos - first_node).unsqueeze(-1)
        dist_first_node = rot_mat.transpose(1, 2) @ dist_first_node
        dist_first_node = dist_first_node.squeeze(-1)
        dist_first_node_norm = dist_first_node.norm(dim=1, keepdim=True)

        return dist_first_node, dist_first_node_norm

    def _compute_prin_feat(self, node_pos):
        """Compute principal axis direction for each rigid body.

        Principal axis is the normalized vector from sphere0 to sphere1 of each rod.

        Args:
            node_pos: [batch_size * num_nodes, 3] node positions

        Returns:
            [batch_size * num_nodes, 3] principal axis direction per node (replicated)
        """
        num_nodes, num_nodes_per_rod = self.robot.num_nodes, self.robot.num_nodes_per_rod

        node_pos_ = node_pos.reshape(-1, 3 * num_nodes_per_rod)
        prin = (node_pos_[:, 3 * self.sphere1_idx: 3 * self.sphere1_idx + 3]
                - node_pos_[:, 3 * self.sphere0_idx: 3 * self.sphere0_idx + 3])
        prin = prin / prin.norm(dim=1, keepdim=True)
        prin = prin.repeat(1, num_nodes_per_rod).reshape(-1, 3)

        return prin

    def _compute_node_feats(self,
                            node_pos: torch.Tensor,
                            prev_node_pos: torch.Tensor,
                            batch_size: int,
                            batch_pos: torch.Tensor,
                            **kwargs
                            ) -> NodeFeats:
        """Compute all node features from current and previous positions.

        Calculates velocities, spatial relationships, mass properties, and geometric
        features. Adds ground node features at the end.

        Args:
            node_pos: [batch_size * num_nodes, 3] current positions
            prev_node_pos: [batch_size * num_nodes, 3] previous positions
            batch_size: Number of graphs in batch
            batch_pos: [batch_size * num_rods, 3] center of mass positions
            **kwargs: Must contain 'dataset_idx' for one-hot encoding

        Returns:
            NodeFeats containing all computed node features
        """

        # Pre adding ground node
        num_nodes = self.robot.num_nodes_per_rod
        com_pos = batch_pos.repeat(1, num_nodes, 1).reshape(-1, 3)
        dist_from_com = node_pos - com_pos
        dist_from_com_norm = dist_from_com.norm(dim=1, keepdim=True)
        dir_from_com = safe_norm(dist_from_com)

        node_vels = (node_pos - prev_node_pos) / self.dt.squeeze(-1)

        dist_to_ground = node_pos[:, 2:3] - self.sphere_radius
        dist_to_ground = torch.clamp_max(dist_to_ground, self.MAX_DIST_TO_GRND)

        dist_first_node, dist_first_node_norm = self._compute_shape_feats(node_pos, batch_size)
        node_prin = self._compute_prin_feat(node_pos)

        # Post ground node
        grnd_ten1 = zeros((1, 3), ref_tensor=node_pos)
        grnd_ten2 = torch.tensor([0., 0., 1.], dtype=self.dtype, device=self.device)

        node_pos = self._inject_grnd_feat(node_pos, grnd_ten1)
        node_vels = self._inject_grnd_feat(node_vels, grnd_ten1)
        dist_from_com_norm = self._inject_grnd_feat(dist_from_com_norm, grnd_ten1[:, :1])
        dir_from_com = self._inject_grnd_feat(dir_from_com, grnd_ten1)
        dist_to_ground = self._inject_grnd_feat(dist_to_ground, grnd_ten1[:, :1])
        dist_first_node = self._inject_grnd_feat(dist_first_node, grnd_ten1)
        dist_first_node_norm = self._inject_grnd_feat(dist_first_node_norm, grnd_ten1[:, :1])
        node_prin = self._inject_grnd_feat(node_prin, grnd_ten2)

        # Need to cache batch size before data processor call
        body_verts = self._feats_batch_cache[batch_size].node_body_verts
        inv_mass = self._feats_batch_cache[batch_size].node_inv_mass
        inv_inertia = self._feats_batch_cache[batch_size].node_inv_inertia
        body_mask = self._feats_batch_cache[batch_size].body_mask

        node_feats = NodeFeats(
            node_vel=node_vels,
            node_inv_mass=inv_mass,
            node_inv_inertia=inv_inertia,
            node_dir_from_com=dir_from_com,
            node_dist_from_com_norm=dist_from_com_norm,
            node_dist_to_ground=dist_to_ground,
            node_body_verts=body_verts,
            node_dist_to_first_node=dist_first_node,
            node_dist_to_first_node_norm=dist_first_node_norm,
            node_pos=node_pos,
            node_prev_pos=prev_node_pos,
            node_prin_axis=node_prin,
            node_sim_type=self._one_hot_encode(kwargs['dataset_idx']),
            body_mask=body_mask
        )

        return node_feats

    def _one_hot_encode(self, batch_idxs):
        """Create one-hot encoding for dataset indices.

        Used to indicate which dataset each sample comes from in multi-dataset training.

        Args:
            batch_idxs: [batch_size, 1] dataset indices

        Returns:
            [batch_size * num_nodes, num_datasets] one-hot encoded vectors
        """
        num_nodes = self.robot.num_nodes + 1
        batch_idxs = batch_idxs.repeat(1, num_nodes).reshape(-1, 1)
        vecs = torch.zeros(
            (batch_idxs.shape[0], self.NUM_DATASETS),
            dtype=self.dtype,
            device=self.device
        )
        vecs[torch.arange(batch_idxs.shape[0], dtype=torch.int), batch_idxs.flatten()] = 1.

        return vecs

    def _body_edge_index(self) -> torch.Tensor:
        """Get edge indices for rigid body connections.

        Returns:
            [2, num_body_edges] edge connectivity for body structure
        """
        senders = self.robot.template_idx[:1].to(self.device)
        receivers = self.robot.template_idx[1:].to(self.device)

        edge_index = torch.vstack([senders, receivers])

        return edge_index

    def _get_cable_edge_idxs(self) -> torch.Tensor:
        """Get edge indices for cable connections.

        Returns:
            [2, num_cable_edges] edge connectivity for cables
        """
        return self.robot.get_cable_edge_idxs().to(self.device)

    def _contact_edge_index(self,
                            grnd_idx: int
                            ) -> torch.Tensor:
        """Get bidirectional edge indices for ground contact.

        Creates edges connecting contact-capable nodes to ground node in both directions.

        Args:
            grnd_idx: Index of ground node in graph

        Returns:
            [2, 2 * num_contact_nodes] bidirectional contact edge connectivity
        """
        senders = torch.tensor([self.robot.get_contact_nodes()],
                               dtype=torch.int,
                               device=self.device)
        receivers = torch.full((1, len(self.robot.get_contact_nodes())),
                               grnd_idx,
                               dtype=torch.int,
                               device=self.device)
        edge_index = torch.vstack([
            torch.hstack([senders, receivers]),
            torch.hstack([receivers, senders])
        ]).detach()

        return edge_index

    def _compute_edge_idxs(self, batch_size) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve cached edge indices for batch size.

        Args:
            batch_size: Batch size to retrieve cached features for

        Returns:
            Tuple of (body_edge_idx, cable_edge_idx, contact_edge_idx)
        """
        # Need to cache batch size before data processor call
        body_edge_idx = self._feats_batch_cache[batch_size].body_edge_idx
        cable_edge_idx = self._feats_batch_cache[batch_size].cable_edge_idx
        contact_edge_idx = self._feats_batch_cache[batch_size].contact_edge_idx

        return body_edge_idx, cable_edge_idx, contact_edge_idx

    def _compute_body_edge_feats(self, body_edge_idx, node_pos, batch_size) -> BodyEdgeFeats:
        """Compute features for rigid body edges.

        Args:
            body_edge_idx: [2, num_edges] body edge connectivity
            node_pos: [num_nodes, 3] node positions
            batch_size: Batch size for retrieving cached rest distances

        Returns:
            BodyEdgeFeats with current and rest distances
        """
        body_dists = node_pos[body_edge_idx[1]] - node_pos[body_edge_idx[0]]
        body_dists_norm = body_dists.norm(dim=1, keepdim=True)

        # Need to cache batch size before data processor call
        body_rest_dists = self._feats_batch_cache[batch_size].body_rest_dist
        body_rest_dists_norm = self._feats_batch_cache[batch_size].body_rest_dist_norm

        body_edge_feats = BodyEdgeFeats(
            body_dist=body_dists,
            body_dist_norm=body_dists_norm,
            body_rest_dist=body_rest_dists,
            body_rest_dist_norm=body_rest_dists_norm
        )
        return body_edge_feats

    def _compute_contact_edge_feats(self, contact_edge_idx, node_pos, node_vels, batch_size) -> ContactEdgeFeats:
        """Compute features for ground contact edges.

        Calculates contact distances, normals, tangents, and relative velocities.

        Args:
            contact_edge_idx: [2, num_edges] contact edge connectivity
            node_pos: [num_nodes, 3] node positions
            node_vels: [num_nodes, 3] node velocities
            batch_size: Batch size for retrieving cached normals

        Returns:
            ContactEdgeFeats with contact geometry and velocities
        """
        n_rods = len(self.robot.rods) * 2
        body_rcvrs = torch.tensor(
            [[-1] * n_rods + [1] * n_rods], device=node_pos.device
        ).repeat(batch_size, 1).reshape(-1, 1)
        contact_dists = (node_pos[contact_edge_idx[1], 2:3] - node_pos[contact_edge_idx[0], 2:3])

        contact_close_mask = contact_dists * body_rcvrs - self.sphere_radius.squeeze(-1) < self.CONTACT_EDGE_THRESHOLD
        contact_dists = contact_dists - body_rcvrs * self.sphere_radius.squeeze(-1)

        # Need to cache batch size before data processor call
        contact_normal = self._feats_batch_cache[batch_size].contact_normal

        contact_rel_vel = node_vels[contact_edge_idx[1], :3] - node_vels[contact_edge_idx[0], :3]
        contact_rel_vel_normal = torch.linalg.vecdot(
            contact_rel_vel,
            contact_normal,
            dim=1
        ).unsqueeze(1)
        contact_tangent = contact_rel_vel - contact_rel_vel_normal * contact_normal
        contact_rel_vel_tangent = contact_tangent.norm(dim=1, keepdim=True)
        contact_rel_vel_tangent = torch.clamp_min(contact_rel_vel_tangent, 1e-8)
        contact_tangent = contact_tangent / contact_rel_vel_tangent

        contact_edge_feats = ContactEdgeFeats(
            contact_dist=contact_dists,
            contact_normal=contact_normal,
            contact_tangent=contact_tangent,
            contact_rel_vel_normal=contact_rel_vel_normal,
            contact_rel_vel_tangent=contact_rel_vel_tangent,
            contact_close_mask=contact_close_mask
        )
        return contact_edge_feats

    def _compute_cable_edge_feats(self, cable_edge_idx, node_pos, node_vels, batch_size, ctrls) -> CableEdgeFeats:
        """Compute features for cable edges.

        Calculates cable distances, directions, velocities, and material properties.
        Handles both control-based and rest-length-based cable input representations.

        Args:
            cable_edge_idx: [2, num_edges] cable edge connectivity
            node_pos: [num_nodes, 3] node positions
            node_vels: [num_nodes, 3] node velocities
            batch_size: Batch size for retrieving cached properties
            ctrls: [batch_size, num_actuated_cables, num_ctrls] control signals

        Returns:
            CableEdgeFeats with cable geometry, dynamics, and control
        """
        cable_dists = node_pos[cable_edge_idx[1]] - node_pos[cable_edge_idx[0]]
        cable_dists_norm = cable_dists.norm(dim=1, keepdim=True)
        cable_dir = cable_dists / cable_dists_norm
        cable_rel_vel = node_vels[cable_edge_idx[1], :3] - node_vels[cable_edge_idx[0], :3]
        cable_rel_vel_norm = torch.linalg.vecdot(
            cable_rel_vel,
            cable_dir,
            dim=1
        ).unsqueeze(1)

        # Need to cache batch size before data processor call
        cable_stiffness = self._feats_batch_cache[batch_size].cable_stiffness
        cable_damping = self._feats_batch_cache[batch_size].cable_damping

        if self.cable_input_type == CableInputType.CTRLS:
            cable_act_rest_lengths = torch.hstack([
                c.rest_length
                for c in self.robot.actuated_cables.values()
            ]).repeat_interleave(2, dim=1)

            num_nonact_cables = len(self.robot.non_actuated_cables)
            num_ctrls = self.NUM_CTRLS_HIST + self.NUM_OUT_STEPS
            nonact_ctrls = zeros((ctrls.shape[0], num_nonact_cables, num_ctrls), ref_tensor=ctrls)
            cable_ctrls = (torch.hstack([ctrls, nonact_ctrls])
                           .repeat_interleave(2, dim=1)
                           .reshape(-1, num_ctrls))
        else:
            cable_act_rest_lengths = self.robot.gnn_rest_lens.repeat_interleave(2, dim=1)
            cable_ctrls = None

        if len(self.robot.non_actuated_cables) > 0:
            cable_non_act_rest_lengths = torch.hstack([
                c.rest_length.repeat(batch_size, 1, cable_act_rest_lengths.shape[-1])
                for c in self.robot.non_actuated_cables.values()
            ]).repeat_interleave(2, dim=1)
        else:
            cable_non_act_rest_lengths = zeros(
                (cable_act_rest_lengths.shape[0], 0),
                ref_tensor=cable_act_rest_lengths
            )

        cable_rest_lengths = torch.hstack([
            cable_act_rest_lengths,
            cable_non_act_rest_lengths
        ]).reshape(-1, self.cable_edge_feat_dict['cable_rest_length'])

        cable_act_mask = self._feats_batch_cache[batch_size].cable_actuated_mask

        cable_edge_feats = CableEdgeFeats(
            cable_dist=cable_dists,
            cable_dist_norm=cable_dists_norm,
            cable_dir=cable_dir,
            cable_rel_vel_norm=cable_rel_vel_norm,
            cable_rest_length=cable_rest_lengths,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            cable_ctrls=cable_ctrls,
            cable_actuated_mask=cable_act_mask
        )
        return cable_edge_feats

    def _compute_edge_feats(self,
                            node_feats,
                            body_edge_idx,
                            cable_edge_idx,
                            contact_edge_idx,
                            batch_size,
                            **kwargs):
        """Compute all edge features for body, cable, and contact edges.

        Args:
            node_feats: NodeFeats containing node positions and velocities
            body_edge_idx: [2, num_body_edges] body edge connectivity
            cable_edge_idx: [2, num_cable_edges] cable edge connectivity
            contact_edge_idx: [2, num_contact_edges] contact edge connectivity
            batch_size: Number of graphs in batch
            **kwargs: Must contain 'ctrls' for cable control signals

        Returns:
            Tuple of (body_edge_feats, cable_edge_feats, contact_edge_feats)
        """
        # body edges
        body_edge_feats = self._compute_body_edge_feats(
            body_edge_idx,
            node_feats.node_pos,
            batch_size
        )

        # contact edges
        contact_edge_feats = self._compute_contact_edge_feats(
            contact_edge_idx,
            node_feats.node_pos,
            node_feats.node_vel,
            batch_size,
        )

        # cable edges
        cable_edge_feats = self._compute_cable_edge_feats(
            cable_edge_idx,
            node_feats.node_pos,
            node_feats.node_vel,
            batch_size,
            ctrls=kwargs['ctrls'],
        )

        return body_edge_feats, cable_edge_feats, contact_edge_feats

    def _get_body_mask(self, batch_size, device):
        """Create boolean mask distinguishing body nodes from ground node.

        Args:
            batch_size: Number of graphs in batch
            device: Target device for tensor

        Returns:
            [batch_size * (num_nodes + 1), 1] mask (True for body, False for ground)
        """
        body_mask = torch.tensor(
            [True] * self.robot.num_nodes + [False],
            dtype=torch.bool,
            device=device
        ).repeat(batch_size, 1).reshape(-1, 1)
        return body_mask

    def forward(self,
                batch_state: torch.Tensor,
                **kwargs: torch.Tensor):
        """Convert batch of SE(3) robot states to graph-structured features.

        Main processing pipeline that transforms rigid body states into complete
        graph representations suitable for GNN physics prediction.

        Args:
            batch_state: [batch_size, 13 * num_rods] SE(3) states
                Format per rod: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            **kwargs: Must contain:
                - 'ctrls': [batch_size, num_cables, num_ctrls] control signals
                - 'dataset_idx': [batch_size, 1] dataset identifiers

        Returns:
            Tuple of (graph_feats, raw_feats):
                - graph_feats: GraphFeats with normalized features ready for GNN
                - raw_feats: Tuple of raw feature NamedTuples for auxiliary tasks
        """
        batch_size = batch_state.shape[0]

        # Convert batch state to node_pos and prev_node_pos
        batch_state_ = batch_state.reshape(-1, 13, 1)
        batch_pos = batch_state_[:, :3]
        batch_quat = batch_state_[:, 3:7]
        batch_lin_vel = batch_state_[:, 7:10]
        batch_ang_vel = batch_state_[:, 10:13]

        batch_prev_pos = batch_state_[:, :3] - self.dt * batch_lin_vel
        batch_prev_quat = torch_quaternion.update_quat(
            batch_state_[:, 3:7], -batch_ang_vel, self.dt
        )

        node_pos = self.pose2node(
            batch_pos, batch_quat, batch_size
        )
        prev_node_pos = self.pose2node(
            batch_prev_pos, batch_prev_quat, batch_size
        )

        # Compute node feats
        node_raw_feats = self._compute_node_feats(
            node_pos,
            prev_node_pos,
            batch_size,
            batch_pos,
            dataset_idx=kwargs['dataset_idx']
        )

        node_hidden_state = zeros(
            (node_raw_feats.node_pos.shape[0], self.node_hidden_state_size),
            ref_tensor=node_raw_feats.node_pos
        )

        # Compute edge indices
        body_edge_idx, cable_edge_idx, contact_edge_idx = self._compute_edge_idxs(batch_size)

        # Compute edge feats
        body_edge_feats, cable_edge_feats, contact_edge_feats = self._compute_edge_feats(
            node_raw_feats,
            body_edge_idx,
            cable_edge_idx,
            contact_edge_idx,
            batch_size,
            ctrls=kwargs['ctrls'],
        )

        node_x, body_edge_attr, cable_edge_attr, contact_edge_attr = (
            self.get_normalize_feats(node_raw_feats, body_edge_feats, cable_edge_feats, contact_edge_feats)
        )

        raw_feats = (node_raw_feats, body_edge_feats, cable_edge_feats, contact_edge_feats)

        graph_feats = GraphFeats(
            node_x=node_x,
            body_edge_idx=body_edge_idx,
            cable_edge_idx=cable_edge_idx,
            contact_edge_idx=contact_edge_idx,
            body_edge_attr=body_edge_attr,
            cable_edge_attr=cable_edge_attr,
            contact_edge_attr=contact_edge_attr,
            contact_close_mask=contact_edge_feats.contact_close_mask,
            node_hidden_state=node_hidden_state
        )

        return graph_feats, raw_feats

    @staticmethod
    def feats2graph(graph_feats: GraphFeats, raw_feats: List):
        """Convert feature NamedTuples to PyTorch Geometric Data object.

        Combines normalized graph features and raw features into a single
        graph data object compatible with PyTorch Geometric.

        Args:
            graph_feats: GraphFeats with normalized features
            raw_feats: List of raw feature NamedTuples

        Returns:
            torch_geometric.data.Data object with all features as attributes
        """
        combined_graph_feats = {
            **graph_feats._asdict(),
            **{k: v for raw_feat in raw_feats for k, v in raw_feat._asdict().items()},
            'cable_edge_index': graph_feats.cable_edge_idx.to(torch.long),
            'contact_edge_index': graph_feats.contact_edge_idx.to(torch.long),
            'body_edge_index': graph_feats.body_edge_idx.to(torch.long),
            'pos': raw_feats[0].node_pos,
            'vel': raw_feats[0].node_vel,
        }
        graph = GraphData(**combined_graph_feats)
        return graph
