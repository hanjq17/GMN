import numpy as np
import torch
import random
import pickle as pkl


class NBodyMStickDataset():
    """
    NBodyDataset

    """
    def __init__(self, partition='train', max_samples=1e8, data_dir='',
                 n_isolated=5, n_stick=0, n_hinge=0):
        self.partition = partition
        self.data_dir = data_dir
        self.n_isolated,  self.n_stick, self.n_hinge = n_isolated, n_stick, n_hinge

        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition

        self.suffix += '_charged{:d}_{:d}_{:d}'.format(n_isolated, n_stick, n_hinge)

        self.max_samples = int(max_samples)
        self.data, self.edges, self.cfg = self.load()

    def load(self):
        loc = np.load(self.data_dir + '/' + 'loc_' + self.suffix + '.npy')
        vel = np.load(self.data_dir + '/' + 'vel_' + self.suffix + '.npy')
        charges = np.load(self.data_dir + '/' + 'charges_' + self.suffix + '.npy')
        edges = np.load(self.data_dir + '/' + 'edges_' + self.suffix + '.npy')
        with open(self.data_dir + '/' + 'cfg_' + self.suffix + '.pkl', 'rb') as f:
            cfg = pkl.load(f)

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges, cfg

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        # convert stick [M, 2]
        loc, vel = torch.Tensor(loc), torch.Tensor(vel)  # remove transpose this time
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0: self.max_samples]
        edges = edges[: self.max_samples, ...]  # add here for better consistency
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # remove self loop
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]

        # swap n_nodes <--> batch_size and add nf dimension
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)  # [B, N*(N-1), 1]

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges, self.cfg = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]
        frame_0, frame_T = 30, 40
        # concat stick indicator to edge_attr (for egnn_vel)
        edges = self.edges
        # initialize the configurations
        cfg = self.cfg[i]
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            if 'Stick' in cfg:
                for stick in cfg['Stick']:
                    if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
                    # if (row == stick[0] and col == stick[1]) or (row == stick[1] and col == stick[0]):
                        stick_ind[m] = 1
            if 'Hinge' in cfg:
                for hinge in cfg['Hinge']:
                    if (row, col) in [(hinge[0], hinge[1]), (hinge[1], hinge[0]), (hinge[0], hinge[2]), (hinge[2], hinge[0])]:
                        stick_ind[m] = 2
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)


        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T], vel[frame_T], cfg

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg
