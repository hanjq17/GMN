import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class GMNLayer(nn.Module):
    def __init__(self, vec_in_dim, scalar_in_dim, edge_in_dim, hidden_dim, vec_out_dim, scalar_out_dim,
                 act_fn, norm_type, n_head):
        super().__init__()
        self.norm_type = norm_type
        self.n_head = n_head
        self.edge_fc = nn.Linear(edge_in_dim, hidden_dim)
        self.vec_in_dim = vec_in_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + ((vec_in_dim // n_head + 1) ** 2) * n_head + scalar_in_dim * 2, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, (vec_in_dim // n_head + 1) * (vec_out_dim // n_head) * n_head)  # + 1 for the radial vector
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_in_dim + hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, scalar_out_dim)
        )

    def forward(self, Z, h, edge_index, edge_distance_feature, edge_distance_vec, edge_distance):
        edge_feature = self.edge_fc(edge_distance_feature)
        h_j, h_i = h[edge_index[0]], h[edge_index[1]]
        Z_j, Z_i = Z[edge_index[0]], Z[edge_index[1]]
        Z_ij = Z_i - Z_j
        # multi-head
        Z_ij = Z_ij.view(-1, Z_ij.size(1), self.vec_in_dim // self.n_head, self.n_head)
        Z_ij = torch.cat((Z_ij, edge_distance_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.n_head)), dim=-2)
        # temp_Z = Z_ij.permute(0, 3, 1, 2)
        # invar = (temp_Z.transpose(-1, -2) @ temp_Z).flatten(1)
        invar = torch.einsum('btdh,bdrh->btrh', Z_ij.transpose(-2, -3), Z_ij).flatten(1)
        # Z_ij = torch.cat((Z_ij, edge_distance_vec.unsqueeze(-1)), dim=-1)
        # invar = Z_ij.transpose(-1, -2) @ Z_ij
        # invar = invar.flatten(-2)
        if self.norm_type == 'pre':
            invar = F.normalize(invar, p=2, dim=-1)
        msg = self.msg_mlp(torch.cat((h_i, h_j, invar, edge_feature), dim=-1))
        if self.norm_type == 'post':
            norm = invar.norm(dim=-1, keepdim=True)
            msg = msg / (norm + 1)
        basis = self.vec_mlp(msg)
        basis = basis.view(Z_ij.size(0), Z_ij.size(2), -1, Z_ij.size(3))
        Z_agg = torch.einsum('bdth,btkh->bdkh', Z_ij, basis).flatten(-2)
        # basis = basis.view(Z_ij.size(0), Z_ij.size(3), Z_ij.size(2), -1)
        # Z_agg = (temp_Z @ basis).permute(0, 2, 3, 1).flatten(-2)
        # basis = basis.view(Z_ij.size(0), Z_ij.size(-1), -1)
        # Z_agg = Z_ij @ basis
        Z_out = scatter(Z_agg, edge_index[1], dim=0, reduce='mean', dim_size=Z.size(0))
        m = scatter(msg, edge_index[1], dim=0, reduce='sum', dim_size=h.size(0))
        h_out = self.scalar_mlp(torch.cat((h, m), dim=-1))
        return Z_out, h_out

