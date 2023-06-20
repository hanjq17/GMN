import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from torch_geometric.nn import global_mean_pool, global_add_pool

class InteractionBlock(nn.Module):
    """ A wrapper for multi-head vector-scalar interaction block with non-linearity.
    The output is invariant."""
    def __init__(self, vec_in_dim, scalar_in_dim, hidden_dim, out_dim, n_head, act_fn, norm_type):
        super().__init__()
        assert vec_in_dim % n_head == 0, 'vec_in_dim not divided by n_head'
        self.net = nn.Sequential(
            nn.Linear(((vec_in_dim // n_head) ** 2) * n_head + scalar_in_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.norm_type = norm_type

    def forward(self, vec, scalar):
        """
        :param vec: [BN, 3, vH], geometric vector.
        :param scalar: [BN, H], features.
        :return: [BN, H_out], the invariant output.
        """
        invar = torch.einsum('btdh,bdrh->btrh', vec.transpose(-2, -3), vec).flatten(1) # [BN, vH/head, vH/head, head]
        if self.norm_type == 'pre':
            invar = F.normalize(invar, p=2, dim=-1)
        invar_ = torch.cat((invar, scalar), dim=-1)  # [BN, vH/head * vH/head * head + H]
        out = self.net(invar_)  # [BN, vH_out/head * vH_out'/head * head + H_out*H_out']
        if self.norm_type == 'post':
            norm = invar.norm(p=2, dim=-1, keepdim=True)  # [BM, 1]
            out = out / (norm + 1)
        return out  # [BN, H_out]


class Filter(nn.Module):
    """
    The filter layer which accounts for field message passing between node i and j.
    The filter is O(3)-invariant w.r.t. Z_i, for an arbitrary node i.
    """
    def __init__(self, vec_in_dim, scalar_in_dim, hidden_dim,
                 vec_out_dim, vec_out_dim_, scalar_out_dim, act_fn,
                 norm_type, n_head):
        super().__init__()
        self.block = InteractionBlock(vec_in_dim, 2 * scalar_in_dim, hidden_dim,
                                      (vec_out_dim // n_head) * (vec_out_dim_ // n_head) * n_head + scalar_out_dim,
                                      n_head, act_fn, norm_type)
        self.vec_in_dim, self.scalar_in_dim = vec_in_dim, scalar_in_dim
        self.vec_out_dim, self.vec_out_dim_ = vec_out_dim, vec_out_dim_
        self.scalar_out_dim = scalar_out_dim
        self.n_head = n_head

    def forward(self, Z_ij, h_i, h_j):
        """
        :param Z_ij: [BM, 3, vH].
        :param h_i: [BM, H].
        :param h_j: [BM, H].
        :return: The filter values, vector filters: [BM, vH_out*vH_out'], scalars: [BM, H_out].
        """
        # invariant, multi-head
        Z_ij = Z_ij.view(-1, Z_ij.size(1), self.vec_in_dim // self.n_head, self.n_head)
        h_ij = torch.cat((h_i, h_j), dim=-1)
        out = self.block(Z_ij, h_ij)
        v_out, h_out = out[..., :-self.scalar_out_dim], out[..., -self.scalar_out_dim:]
        v_out = v_out.view(-1, self.vec_out_dim // self.n_head, self.vec_out_dim_ // self.n_head, self.n_head)
        return v_out, h_out  # [BM, vH_out/head, vH_out'/head, head]， [BM, H_out]


class BasisFieldFunc(nn.Module):
    """
    The Basis of Field Functions.
    It is O(3)-equivariant w.r.t. Z_i, but has no restrictions on r_i, for an arbitrary node i.
    """
    def __init__(self, vec_in_dim, scalar_in_dim, hidden_dim,
                 vec_out_dim, scalar_out_dim, act_fn, norm_type, n_head, learnable_frame, sampling_scope,
                 num_gaussians, distance_expansion, local_radius_cutoff, concat_r_to_z, separate_head):
        super().__init__()
        r_in_num = 1  # Independent sampling
        self.r_in_num = r_in_num

        self.vec_in_dim, self.scalar_in_dim = vec_in_dim, scalar_in_dim
        self.hidden_dim = hidden_dim
        self.vec_out_dim, self.scalar_out_dim = vec_out_dim, scalar_out_dim
        self.norm_type = norm_type
        self.n_head = n_head
        self.learnable_frame = learnable_frame
        self.sampling_scope = sampling_scope
        self.distance_expansion = distance_expansion
        self.local_radius_cutoff = local_radius_cutoff
        self.concat_r_to_z = concat_r_to_z
        self.separate_head = separate_head

        if not self.learnable_frame:
            if not concat_r_to_z:
                self.block = InteractionBlock(vec_in_dim, scalar_in_dim + r_in_num * 3 + num_gaussians, hidden_dim,
                                              (vec_in_dim // n_head) * (vec_out_dim // n_head) * n_head + scalar_out_dim,
                                              n_head, act_fn, norm_type)
            else:
                # Indeed, one can also concat r_direction to the vector channel, along with Z
                _vec_in_dim = vec_in_dim + r_in_num * n_head
                self.block = InteractionBlock(_vec_in_dim, scalar_in_dim + num_gaussians, hidden_dim,
                                              (_vec_in_dim // n_head) * (vec_out_dim // n_head) * n_head + scalar_out_dim,
                                              n_head, act_fn, norm_type)
        else:
            self.frame_block = InteractionBlock(vec_in_dim, scalar_in_dim, hidden_dim,
                                                vec_in_dim // n_head * 3 * r_in_num * n_head,
                                                n_head, act_fn, norm_type)
            _vec_in_dim = vec_in_dim + r_in_num * n_head
            self.block = InteractionBlock(_vec_in_dim, scalar_in_dim, hidden_dim,
                                          (_vec_in_dim // n_head) * (vec_out_dim // n_head) * n_head + scalar_out_dim,
                                          n_head, act_fn, norm_type)

    def sampling(self, r, data):
        """
        Conduct local or global sampling.
        :param r: [n_r, 3], the pre-sampled points within a **unit** ball.
        :param data: The data object.
        :return: [BN*n_r, 3], the sampled local and global r for each node.
        """
        r_local, r_global = None, None
        pos = data.pos
        if self.sampling_scope in ['local', 'both']:
            r_local = r.unsqueeze(0).expand(pos.size(0), -1, -1) * self.local_radius_cutoff  # [BN, n_r, 3]
            r_local = r_local.flatten(0, 1)  # [BN*n_r, 3]
        if self.sampling_scope in ['global', 'both']:
            center = global_mean_pool(pos, data.batch)  # [B, 3]
            dist_to_center = (pos - center[data.batch]).norm(p=2, dim=-1)  # [BN]
            max_dist_to_center = scatter_max(dist_to_center, data.batch)[0].unsqueeze(1)  # [B, 1]
            r_global = max_dist_to_center.unsqueeze(1) * r.unsqueeze(0) + center.unsqueeze(1)  # [B, n_r, 3]
            r_global = r_global[data.batch]  # [BN, n_r, 3]
            r_global = r_global - pos.unsqueeze(1)
            r_global = r_global.flatten(0, 1)  # [BN*n_r, 3]
        if self.learnable_frame:  # Do not use r global if using learnable frame projection
            r_global = r_local
        return r_local, r_global  # [BN*n_r, 3]

    def _forward_on_samples(self, r, Z, h, num_r_samples):
        """
        Compute basis value based on sampled rs. Can be either basis_local or basis_ext (global).
        :param r: [BN*n_r, 3]. The sampled points.
        :param Z: [BN, 3, vH]. The geometric vectors.
        :param h: [BN, H]. The features.
        :param num_r_samples: The number of r samples, i.e., n_r.
        :return: The computed basis values, vector [BN, n_r, 3, vH_out/head, head], scalar [BN, n_r, H_out]
        """
        if not self.learnable_frame:  # Treat r the same as h
            Z = Z.unsqueeze(1).expand(-1, num_r_samples, -1, -1).flatten(0, 1)  # [BN*n_r, 3, vH]
            if self.concat_r_to_z:
                Z = torch.cat((Z, F.normalize(r, p=2, dim=-1).unsqueeze(-1).expand(-1, -1, self.n_head)), dim=-1)
                # [BN*n_r, 3, vH+head]
            Z = Z.view(-1, Z.size(1), Z.size(2) // self.n_head, self.n_head)  # [BN*n_r, 3, vH/head, head]
            h_expand = h.unsqueeze(1).expand(-1, num_r_samples, -1).flatten(0, 1)  # [BN*n_r, H]
            r_feat = self.distance_expansion(r.norm(p=2, dim=-1))  # TODO: This can be precomputed, but only for r_local
            if not self.concat_r_to_z:
                r_direction = F.normalize(r, p=2, dim=-1)
                r_feat = torch.cat((r_feat, r_direction), dim=-1)  # [BN*n_r, H+3]
            h_expand = torch.cat((r_feat, h_expand), dim=-1)  # [BN*n_r, H+3]
        else:  # Treat r as local frame and project it into global frame with a learnable projection
            _Z = Z.view(-1, Z.size(1), Z.size(2) // self.n_head, self.n_head)  # [BN, 3, vH/head, head]
            v_out = self.frame_block(_Z, h).view(
                -1, _Z.size(-2), 3, self.n_head
            )  # [BN, vH/head, 3, head]
            frame = torch.einsum('bdth,btkh->bdkh', _Z, v_out).mean(dim=-1)  # [BN, 3, 3, head] -> [BN, 3, 3]
            # Project r into global frame
            frame = frame.unsqueeze(1).expand(-1, num_r_samples, -1, -1).flatten(0, 1)  # [BN*n_r, 3, 3]
            r_g = frame @ r.unsqueeze(-1)  # [BN*n_r, 3, 1]
            Z = Z.unsqueeze(1).expand(-1, num_r_samples, -1, -1).flatten(0, 1)  # [BN*n_r, 3, vH]
            Z = torch.cat((Z, r_g.expand(-1, -1, self.n_head)), dim=-1)  # [BN*n_r, 3, vH+head]
            Z = Z.view(-1, Z.size(1), Z.size(2) // self.n_head, self.n_head)  # [BN*n_r, 3, (vH+head)/head, head]
            h_expand = h.unsqueeze(1).expand(-1, num_r_samples, -1).flatten(0, 1)  # [BN*n_r, H]

        out = self.block(Z, h_expand)
        v_out, h_out = out[..., :-self.scalar_out_dim], out[..., -self.scalar_out_dim:]
        v_out = v_out.view(
            -1, Z.size(-2), self.vec_out_dim // self.n_head, self.n_head
        )  # [BN*n_r, (vH+head)/head, vH_out/head, head]
        # equivariant
        v_out = torch.einsum('bdth,btkh->bdkh', Z, v_out)  # [BN*n_r, 3, vH_out/head, head]
        v_out = v_out.contiguous().view(-1, num_r_samples, v_out.size(-3), v_out.size(-2), v_out.size(-1))
        # [BN, n_r, 3, vH_out/head, head]
        h_out = h_out.contiguous().view(-1, num_r_samples, h_out.size(-1))  # [BN, n_r, H_out]
        return v_out, h_out  # [BN, n_r, 3, vH_out/head, head], [BN, n_r, H_out]

    def forward(self, r, Z, h, data):
        """
        :param r: [n_r, 3], where n_r is the number of r samples.
        :param Z: [BN, 3, vH].
        :param h: [BN, H].
        :param data: The data object containing pos and batch index, necessary for global sampling.
        :return: The basis function values, vectors: [BN, n_r, 3, vH_out], scalars: [BN, n_r, H_out].
        """
        # Conduct sampling
        num_r_samples = r.size(0)
        r_local, r_global = self.sampling(r, data)  # [BN*n_r, 3]
        v_out_local, h_out_local, v_out_global, h_out_global = 0, 0, 0, 0

        # Compute filter based on local sampling
        if self.sampling_scope in ['local', 'both']:
            v_out_local, h_out_local = self._forward_on_samples(r_local, Z, h, num_r_samples)
            # if hasattr(self, 'visualize'):
            #     visualize_basis(r_local, v_out_local, h_out_local, num_r_samples, data)
            #     exit(0)

        # Compute filter based on global sampling
        if self.sampling_scope in ['global', 'both']:
            # Pool Z and h
            Z_global = global_mean_pool(Z, data.batch)[data.batch]  # [BN, 3, vH]
            h_global = global_mean_pool(h, data.batch)[data.batch]  # [BN, H]
            v_out_global, h_out_global = self._forward_on_samples(r_global, Z_global, h_global, num_r_samples)

        if not self.separate_head:
            v_out = v_out_local + v_out_global
            h_out = h_out_local + h_out_global
        else:
            # Cat local and global.
            v_out = torch.cat([v_out_local, v_out_global], dim=1)   # [BN, n_l + n_g, 2, vH_out/head, Head]
            h_out = torch.cat([h_out_local, h_out_global], dim=1)   # [BN, n_l + n_g, H_out]
        return v_out, h_out  # [BN, n_r, 3, vH_out/head, head], [BN, n_r, H_out]


class GMNOCLayer(nn.Module):
    """
    The Field Interaction Network layer.
    """
    def __init__(self, vec_in_dim, scalar_in_dim, hidden_dim,
                 vec_out_dim, vec_out_dim_, scalar_out_dim, scalar_out_dim_,
                 act_fn, norm_type, n_head, combine_type, learnable_frame, sampling_scope,
                 num_gaussians, distance_expansion, local_radius_cutoff,
                 concat_r_to_z, basis_shared, separate_head):
        super().__init__()
        if basis_shared:
            self.basis_field_func = basis_shared
        else:
            self.basis_field_func = BasisFieldFunc(vec_in_dim, scalar_in_dim, hidden_dim,
                                                   vec_out_dim, scalar_out_dim, act_fn,
                                                   norm_type, n_head, learnable_frame, sampling_scope,
                                                   num_gaussians, distance_expansion, local_radius_cutoff,
                                                   concat_r_to_z, separate_head)
        self.filter = Filter(vec_in_dim, scalar_in_dim, hidden_dim,
                             vec_out_dim, vec_out_dim_, scalar_out_dim,
                             act_fn, norm_type, n_head)
        if combine_type == 'mul':
            self.scalar_combine = lambda x, y: x * y
        elif combine_type == 'mlp':
            self.combine_net = nn.Sequential(
                nn.Linear(scalar_out_dim + scalar_out_dim, hidden_dim),
                act_fn(),
                nn.Linear(hidden_dim, scalar_out_dim_)
            )
            self.scalar_combine = lambda x, y: self.combine_net(torch.cat((x, y), dim=-1))
        else:
            raise NotImplementedError()
        self.n_head = n_head

    def forward(self, r, Z, h, edge_index, data):
        """
        :param r: [n_r, 3], the pre-sampled points within a **unit** ball.
        :param Z: [BN, 3, vH], the geometric vectors.
        :param h: [BN, H], the features.
        :param edge_index: [2, BM], the edges.
        :param data: The data object.
        :return: The computed basis value for each node and each sampled point, [BN, n_r, 3, vH_out'], [BN, n_r, H_out']
        """
        Z_j, Z_i = Z[edge_index[0]], Z[edge_index[1]]  # [BM, 3, vH]
        h_j, h_i = h[edge_index[0]], h[edge_index[1]]  # [BM, H]
        Z_ij = Z_i - Z_j  # Can devise more sophisticated pairwise aggregation here.
        # \phi(r, Z_i, h_i) equivariant
        vec_basis, scalar_basis = self.basis_field_func(r, Z, h, data)  # [BN, n_r, 3, vH_out/head, head], [BN, n_r, H_out]
        # w(Z_ij, h_i, h_j) invariant
        vec_filter, scalar_filter = self.filter(Z_ij, h_i, h_j)  # [BM, vH_out/head * vH_out'/head], [BM, H_out]
        # \sum_j w(Z_ij, h_i, h_j)
        vec_filter_out = scatter(vec_filter, edge_index[1], dim=0, reduce='mean', dim_size=vec_basis.size(0))
        # \sum_j w(Z_ij, h_i, h_j)
        scalar_filter_out = scatter(scalar_filter, edge_index[1], dim=0, reduce='sum', dim_size=scalar_basis.size(0))
        # Linear combination of vector basis and filter values.
        # \phi(r, Z_i, h_i) * \sum_j w(Z_ij, h_i, h_j)
        vec_out = torch.einsum('brijh,bjkh->brikh', vec_basis, vec_filter_out).contiguous().flatten(-2)
        # [BN, n_r, 3, vH_out'/head, head] -> [BN, n_r, 3, vH_out']
        # Currently use MLP aggregation for scalar basis and filter values. Can also use multiplication like vectors.
        scalar_filter_out = scalar_filter_out.unsqueeze(1).expand(-1, scalar_basis.size(1), -1).flatten(0, 1)
        # MLP(\phi(r, Z_i, h_i), \sum_j w(Z_ij, h_i, h_j))
        scalar_out = self.scalar_combine(scalar_basis.flatten(0, 1), scalar_filter_out)
        scalar_out = scalar_out.view(scalar_basis.size(0), scalar_basis.size(1), -1)  # [BN, n_r, H_out']
        return vec_out, scalar_out  # [BN, n_r, 3, vH_out'], [BN, n_r, H_out']


def visualize_basis(r, vec, scalar, num_r_samples, data):
    """
    Visualize the basis value at given sampled points.
    :param r: [BN*n_r, 3], the sampled points.
    :param vec: [BN, n_r, 3, vH_out/head, head], the vector basis value.
    :param scalar: [BN, n_r, H_out], the scalar basis value.
    :param num_r_samples: n_r.
    :param data: The data object.
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    import numpy as np
    r = r.view(-1, num_r_samples, r.size(-1))  # [BN, n_r, 3]
    vec = vec.flatten(-2)  # [BN, n_r, 3, vH]
    radius_k = 4

    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer):  # or draw
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)

    def plot_node(idx):
        k = 0  # The k idx: which sphere to plot
        a_i = data.atomic_numbers[idx].detach().cpu().numpy()
        print(a_i)
        r_i, scalar_i, vec_i = r[idx], scalar[idx], vec[idx]
        r_i = r_i.view(radius_k, -1, r_i.size(-1))  # [k, n, 3]
        scalar_i = scalar_i.view(radius_k, -1, scalar_i.size(-1)).mean(dim=-1)  # [k, n, H_out] -> [k, n]
        vec_i = vec_i.view(radius_k, -1, vec_i.size(-2), vec_i.size(-1)).mean(dim=-1)  # [k, n, 3, vH] -> [k, n, 3]
        r_i = r_i[k].detach().cpu().numpy()  # [n, 3] on a sphere, select the smallest |r|
        scalar_i = scalar_i[k].squeeze(-1).detach().cpu().numpy()  # [n]
        max_norm = vec_i[k].norm(dim=-1, p=2).max()
        print(vec_i[k].norm(dim=-1, p=2).max().item(), vec_i[k].norm(dim=-1, p=2).min().item())
        vec_i = (vec_i[k] / max_norm).detach().cpu().numpy() * 0.5  # [n, 3], norm = 0.3

        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')

        # plot scalar
        temp = ax.scatter(r_i[..., 0], r_i[..., 1], r_i[..., 2], c=scalar_i)
        fig.colorbar(temp, pad=0.2)

        # plot vector
        for j in range(vec_i.shape[0]):
            a = Arrow3D([r_i[j, 0], r_i[j, 0] + vec_i[j, 0]],
                        [r_i[j, 1], r_i[j, 1] + vec_i[j, 1]],
                        [r_i[j, 2], r_i[j, 2] + vec_i[j, 2]], mutation_scale=4,
                        lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(a)

        save_path = f'/data/ocp/figs/{idx}.pdf'
        plt.title(f'atom number {a_i}')
        plt.savefig(save_path)
        print(f'finished for node {idx}, saved to {save_path}')

    for i in range(data.batch.size(0)):
        plot_node(i)
