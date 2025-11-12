from typing import Dict, List

import torch

from state_objects.composite_body import CompositeBody
from utilities.misc_utils import DEFAULT_DTYPE


class SystemTopology(torch.nn.Module):
    """
    Class to graph of rod and spring attachments.
    Topology dict -> (site name) : [..., Object name, ...]
    Site dict -> (site name) : (world frame coordinate)
    """

    def __init__(self, rigid_bodies, cables, sites_dict: Dict = None):
        """
        :param sites_dict: dictionary of sites and locations
        :param topology: dictionary of sites to associated object ids
        """
        super().__init__()
        self.sites_dict = sites_dict if sites_dict else {}
        self.topology = self._build_topology_map(rigid_bodies, cables)

    def to(self, device):
        if self.sites_dict:
            self.sites_dict = {k: v.to(device) for k, v in self.sites_dict.items()}

        return self

    @staticmethod
    def _build_topology_map(rigid_bodies, cables):
        topology_map = {}
        for body in rigid_bodies:
            sites = body.sites
            for site in sites:
                if site not in topology_map:
                    topology_map[site] = []
                topology_map[site].append(body)

        for cable in cables:
            sites = cable.end_pts
            for site in sites:
                if site not in topology_map:
                    topology_map[site] = []
                topology_map[site].append(cable)

        return topology_map

    def update_site(self, site: str, world_frame_pos: torch.Tensor) -> None:
        """
        Method to update site dict

        :param site: Site name
        :param world_frame_pos: world frame coordinate of site
        """
        self.sites_dict[site] = world_frame_pos

        if site not in self.topology:
            self.topology[site] = []



