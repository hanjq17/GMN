import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from torch_geometric.nn import global_mean_pool, global_add_pool
from ocpmodels.common.registry import registry
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import (
    get_pbc_distances,
    radius_graph_pbc,
)
from .layer import GMNOCLayer
from .smearing import GaussianSmearing
from .gmn_temp import GMNLayer
from dataclasses import dataclass

@dataclass
class GMNOCConfig:
    n_layers: int
    backbone: str
    num_gaussians: int
    basis_width_scalar: float
    hidden_dim: int
    vec_dim: int
    otf_graph: bool
    use_pbc: bool
    regress_forces: bool
    max_num_elements: int
    cutoff: float
    max_num_neighbors: int
    num_r_samples: int
    radius_k: int
    norm_type: str
    residual: bool
    n_head: int
    disable_field: bool
    combine_type: str
    learnable_frame: bool
    sampling_scope: str
    local_radius_cutoff: float
    concat_r_to_z: bool
    force_head: str
    # These three are the configs for graphormer-like position head, irrelevant to GMNOC
    num_kernel: int
    attention_heads: int
    enable_edge_types: bool
    return_mask: bool = False
    share_basis: bool = False
    separate_head: bool = False
    recycle_num: int = 1



@registry.register_model("gmnoc")
class GMNOC(BaseModel):
    """
    The GMN-OC model.
    """
    def __init__(self, num_atoms, bond_feat_dim, num_targets, **kwargs):
        super().__init__(num_atoms, bond_feat_dim, num_targets)
        config = GMNOCConfig(**kwargs)
        self.config = config
        # Initialize configurations
        self.act_fn = nn.SiLU
        self.distance_expansion = GaussianSmearing(
            0.0,
            config.cutoff,
            config.num_gaussians,
            config.basis_width_scalar,
        )
        self.atom_embedding = nn.Embedding(config.max_num_elements, config.hidden_dim)
        self.tag_embedding = nn.Embedding(3, config.hidden_dim)

        # Initialize backbone
        self.backbone = config.backbone
       
        self.backbone_layers = nn.ModuleList()
        for i in range(config.n_layers):
            if config.backbone == 'GMN':
                backbone_layer = GMNLayer(
                    config.vec_dim * 2, config.hidden_dim * 2,
                    config.num_gaussians, config.hidden_dim,
                    config.vec_dim, config.hidden_dim, self.act_fn,
                    config.norm_type, config.n_head
                )
            else:
                raise NotImplementedError('Not implemented backbone', config.backbone)
            self.backbone_layers.append(backbone_layer)
        # Initialize field networks
        self.field_layers = nn.ModuleList()
        if not config.disable_field:
            for i in range(config.n_layers + 1):
                layer = GMNOCLayer(
                    config.vec_dim, config.hidden_dim, config.hidden_dim,
                    config.vec_dim, config.vec_dim, config.hidden_dim, config.hidden_dim, self.act_fn,
                    config.norm_type, config.n_head, config.combine_type, config.learnable_frame,
                    config.sampling_scope, config.num_gaussians, self.distance_expansion,
                    config.local_radius_cutoff, config.concat_r_to_z,
                    basis_shared=self.field_layers[0].basis_field_func if config.share_basis and i > 0 else None,
                    separate_head=config.separate_head
                )
                self.field_layers.append(layer)
        # Initialize output head
        if not config.separate_head:
            self.energy_head = nn.Sequential(
                nn.Linear(config.hidden_dim + config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, 1)
            )
        else:
            self.energy_head_scalar_local = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, 1)
            )
            self.energy_head_scalar_global = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, 1)
            )
            self.energy_head_h = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, 1)
            )

        if config.regress_forces and config.force_head == 'linear':
            self.force_head = nn.Linear(config.vec_dim + config.vec_dim, 1, bias=False)
        elif config.regress_forces and config.force_head == 'combine':
            self.force_head = nn.Sequential(
                nn.Linear(config.hidden_dim + config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self.act_fn(),
                nn.Linear(config.hidden_dim, config.vec_dim + config.vec_dim)
            )
        else:
            self.force_head = None

        self.r = torch.zeros(config.num_r_samples, 3)
        print(self)

    def prepare_graph(self, data):
        pos = data.pos
        if self.config.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(data, self.config.cutoff,
                                                                   self.config.max_num_neighbors)
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
        if self.config.use_pbc:
            # assert atomic_numbers.dim() == 1 and atomic_numbers.dtype == torch.long
            out = get_pbc_distances(pos, data.edge_index, data.cell, data.cell_offsets,
                                    data.neighbors, return_distance_vec=True)
            edge_index = out["edge_index"]
            edge_distance_vec = out["distance_vec"]
            edge_distance = out["distances"]
        else:
            edge_index = radius_graph(pos, r=self.config.cutoff, batch=data.batch)
            edge_distance_vec = pos[edge_index[0]] - pos[edge_index[1]]
            edge_distance = edge_distance_vec.norm(dim=-1)
        return data, edge_index, edge_distance_vec, edge_distance

    def forward(self, data):
        # Construct pbc graph
        atomic_numbers = data.atomic_numbers.long()
        data, edge_index, edge_distance_vec, edge_distance = self.prepare_graph(data)
        edge_distance_feature = self.distance_expansion(edge_distance)
        # Initialize Z and h
        h = self.atom_embedding(atomic_numbers)  # [BN, H]
        h = h + self.tag_embedding(data.tags)  # [BN, H]
        Z = torch.zeros(h.size(0), 3, self.config.vec_dim).to(h.device)  # [BN, 3, vH]
        # Prepare r
        r = self.r.to(data.pos.device)  # [n_r, 3]
        # Forward on field network
        integral_vec, integral_scalar = 0, 0

        n_layers = self.config.n_layers + 1
        for i in range(n_layers):
            if i >= 1:
                Z_concat = torch.cat((Z, integral_vec), dim=-1)
                h_concat = torch.cat((h, integral_scalar), dim=-1)
                dZ, dh = self.backbone_layers[i - 1](Z_concat, h_concat, edge_index, edge_distance_feature,
                                                     edge_distance_vec, edge_distance)
                (Z, h) = (Z + dZ, h + dh) if self.config.residual else (dZ, dh)
            if not self.config.disable_field:  # Use field net
                cur_field_vec, cur_field_scalar = self.field_layers[i](r, Z, h, edge_index, data)
            else:
                cur_field_vec = torch.ones_like(Z).unsqueeze(1).expand(-1, r.size(1), -1, -1)
                cur_field_scalar = torch.ones_like(h).unsqueeze(1).expand(-1, r.size(1), -1)
            # Compute the integral
            d_integral_vec = cur_field_vec.mean(dim=1)  # [BN, 3, vH_out']
            d_integral_scalar = cur_field_scalar.mean(dim=1)  # [BN, H_out']
            if not self.config.separate_head:
                # d_integral_vec = global_mean_pool(d_integral_vec, data.batch)[data.batch]
                # d_integral_scalar = global_mean_pool(d_integral_scalar, data.batch)[data.batch]
                (integral_vec, integral_scalar) = (
                    integral_vec + d_integral_vec, integral_scalar + d_integral_scalar) if self.config.residual else (
                    d_integral_vec, d_integral_scalar)
            else:
                if i == n_layers - 1:  # Try the sum output here.
                    # Last layer.
                    integral_vec = d_integral_vec + integral_vec if self.config.residual else d_integral_vec  # [BN, n_r, H_out']
                    integral_scalar = cur_field_scalar  # [BN, n_r, H_out']
                else:
                    (integral_vec, integral_scalar) = (
                        integral_vec + d_integral_vec, integral_scalar + d_integral_scalar) if self.config.residual else (
                        d_integral_vec, d_integral_scalar)

        if not self.config.separate_head:
            # Aggregate each layer's outputs
            vec, scalar = torch.cat((Z, integral_vec), dim=-1), torch.cat((h, integral_scalar), dim=-1)

            # Derive energy and force
            energy = self.energy_head(scalar)
            output_mask = data['tags'] > 0
            energy = energy * output_mask.unsqueeze(-1)
            energy = global_add_pool(energy, data.batch)  # [B, 1]
        else:
            # Aggregate each layer's outputs
            vec = torch.cat((Z, integral_vec), dim=-1)
            scalar = torch.cat((h, integral_scalar.mean(dim=1)), dim=-1)
            local_scalar_integral = integral_scalar[:, :r.size(0), :].sum(dim=1)  # [BN,n_r,  vH]
            # Another half of n_r is also updated locally,
            global_scalar_integral = integral_scalar  # [BN, 2*n_r, vH]
            scalar_h = h

            # Derive energy and force
            energy = self.energy_head_h(scalar_h) + self.energy_head_scalar_local(local_scalar_integral)
            energy_ext = self.energy_head_scalar_global(global_scalar_integral)  # [NB, 2 * n_r, 1]
            output_mask = data['tags'] > 0
            energy = energy * output_mask.unsqueeze(-1)
            energy = global_add_pool(energy, data.batch)  # [B, 1]
            energy_ext = global_mean_pool(energy_ext, data.batch).mean(dim=1)  # [B, 1]
            energy = energy + 0.1 * energy_ext

        if self.config.regress_forces and self.config.force_head == 'linear':
            force = self.force_head(vec).squeeze(-1)  # [BN, 3]
            output = (energy, force, output_mask) if self.config.return_mask else (energy, force)
        elif self.config.regress_forces and self.config.force_head == 'combine':
            force = (self.force_head(scalar).unsqueeze(1) * vec).mean(dim=-1)
            output = (energy, force, output_mask) if self.config.return_mask else (energy, force)
        else:
            output = energy

        return output

