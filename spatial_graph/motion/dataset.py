import numpy as np
import torch
import pickle as pkl
import os


class MotionDataset():
    """
    Motion Dataset

    """

    def __init__(self, partition, max_samples, delta_frame, data_dir):
        with open(os.path.join(data_dir, 'motion.pkl'), 'rb') as f:
            edges, X = pkl.load(f)

        V = []
        for i in range(len(X)):
            V.append(X[i][1:] - X[i][:-1])
            X[i] = X[i][:-1]


        N = X[0].shape[1]

        train_case_id = [20, 1, 17, 13, 14, 9, 4, 2, 7, 5, 16]
        val_case_id = [3, 8, 11, 12, 15, 18]
        test_case_id = [6, 19, 21, 0, 22, 10]


        split_dir = os.path.join(data_dir, 'split.pkl')

        self.partition = partition

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except:
            np.random.seed(100)

            # sample 100 for each case
            itv = 300
            train_mapping = {}
            for i in train_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=100, replace=False)
                train_mapping[i] = sampled
            val_mapping = {}
            for i in val_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=100, replace=False)
                val_mapping[i] = sampled
            test_mapping = {}
            for i in test_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=100, replace=False)
                test_mapping[i] = sampled

            with open(split_dir, 'wb') as f:
                pkl.dump((train_mapping, val_mapping, test_mapping), f)

            print('Generate and save split!')
            split = (train_mapping, val_mapping, test_mapping)

        if partition == 'train':
            mapping = split[0]
        elif partition == 'val':
            mapping = split[1]
        elif partition == 'test':
            mapping = split[2]
        else:
            raise NotImplementedError()

        each_len = max_samples // len(mapping)

        x_0, v_0, x_t, v_t = [], [], [], []
        for i in mapping:
            st = mapping[i][:each_len]
            cur_x_0 = X[i][st]
            cur_v_0 = V[i][st]
            cur_x_t = X[i][st + delta_frame]
            cur_v_t = V[i][st + delta_frame]
            x_0.append(cur_x_0)
            v_0.append(cur_v_0)
            x_t.append(cur_x_t)
            v_t.append(cur_v_t)
        x_0 = np.concatenate(x_0, axis=0)
        v_0 = np.concatenate(v_0, axis=0)
        x_t = np.concatenate(x_t, axis=0)
        v_t = np.concatenate(v_t, axis=0)

        print('Got {:d} samples!'.format(x_0.shape[0]))

        self.n_node = N

        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                        assert not self.atom_edge2[i][j]
                    if self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([2])
                        assert not self.atom_edge[i][j]

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = edges  # [2, edge]

        self.x_0, self.v_0, self.x_t, self.v_t = torch.Tensor(x_0), torch.Tensor(v_0), torch.Tensor(x_t), torch.Tensor(
            v_t)
        mole_idx = np.ones(N)
        self.mole_idx = torch.Tensor(mole_idx)

        self.cfg = self.sample_cfg()

    def sample_cfg(self):
        """
        Kinematics Decomposition
        """
        cfg = {}

        cfg['Stick'] = [(0, 11), (12, 13)]
        cfg['Stick'].extend([(2, 3), (7, 8), (17, 18), (24, 25)])

        cur_selected = []
        for _ in cfg['Stick']:
            cur_selected.append(_[0])
            cur_selected.append(_[1])

        cfg['Isolated'] = [[_] for _ in range(self.n_node) if _ not in cur_selected]
        if len(cfg['Isolated']) == 0:
            cfg.pop('Isolated')

        return cfg

    def __getitem__(self, i):

        cfg = self.cfg

        edge_attr = self.edge_attr
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        edges = self.edges

        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            if 'Stick' in cfg:
                for stick in cfg['Stick']:
                    if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
                        stick_ind[m] = 1
            if 'Hinge' in cfg:
                for hinge in cfg['Hinge']:
                    if (row, col) in [(hinge[0], hinge[1]), (hinge[1], hinge[0]), (hinge[0], hinge[2]), (hinge[2], hinge[0])]:
                        stick_ind[m] = 2
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)  # [edge, 2]
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}

        return self.x_0[i], self.v_0[i], edge_attr, self.mole_idx.unsqueeze(-1), self.x_t[i], self.v_t[i], self.mole_idx.unsqueeze(-1), cfg

    def __len__(self):
        return len(self.x_0)

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


if __name__ == '__main__':
    data = MotionDataset(partition='train', max_samples=500, delta_frame=30, data_dir='')