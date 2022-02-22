import numpy as np
from physical_objects import Isolated, Stick, Hinge
from tqdm import tqdm


class System:
    def __init__(self, n_isolated, n_stick, n_hinge, delta_t=0.001,
                 box_size=None, loc_std=1., vel_norm=0.5,
                 interaction_strength=1., charge_types=None,
                 ):
        self.n_isolated, n_stick, n_hinge = n_isolated, n_stick, n_hinge
        self.delta_t = delta_t
        self._max_F = 0.1 / self.delta_t  # tentative setting
        self.box_size = box_size
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.dim = 3

        self.n_balls = n_isolated * 1 + n_stick * 2 + n_hinge * 3
        n = self.n_balls
        self.loc_std = loc_std * (float(self.n_balls) / 5.) ** (1 / 3) + 0.1

        if charge_types is None:
            charge_types = [1.0, -1.0]
        self.charge_types = charge_types

        self.diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(self.diag_mask, 0)

        charges = np.random.choice(self.charge_types, size=(self.n_balls, 1))
        self.charges = charges
        edges = charges.dot(charges.transpose())
        self.edges = edges

        # Initialize location and velocity
        X = np.random.randn(n, self.dim) * self.loc_std  # N(0, loc_std)
        V = np.random.randn(n, self.dim)  # N(0, 1)
        v_norm = np.sqrt((V ** 2).sum(axis=-1)).reshape(-1, 1)
        V = V / v_norm * self.vel_norm

        # initialize physical objects
        self.physical_objects = []
        # node_idx = 0
        selected = []
        for _ in range(n_isolated):
            rest = [idx for idx in range(self.n_balls) if idx not in selected]
            node_idx = list(np.random.choice(rest, size=1, replace=False))
            current_obj = Isolated(n_balls=1, node_idx=node_idx,
                                   charge=[charges[node_idx[0]]], type='Isolated')
            selected.extend(node_idx)
            self.physical_objects.append(current_obj)

        for _ in range(n_stick):
            rest = [idx for idx in range(self.n_balls) if idx not in selected]
            node_idx = list(np.random.choice(rest, size=2, replace=False))
            current_obj = Stick(n_balls=2, node_idx=node_idx,
                                charge=[charges[node_idx[0]], charges[node_idx[1]]], type='Stick')
            selected.extend(node_idx)
            self.physical_objects.append(current_obj)

        for _ in range(n_hinge):
            rest = [idx for idx in range(self.n_balls) if idx not in selected]
            node_idx = list(np.random.choice(rest, size=3, replace=False))
            current_obj = Hinge(n_balls=3, node_idx=node_idx,
                                charge=[charges[node_idx[0]], charges[node_idx[1]], charges[node_idx[2]]], type='Hinge')
            selected.extend(node_idx)
            self.physical_objects.append(current_obj)

        assert n == self.n_balls == len(selected) == len(np.unique(selected))

        # check and adjust initial conditions
        for obj in self.physical_objects:
            X, V = obj.initialize(X, V)

        # book-keeping x and v
        self.X, self.V = X, V

    @staticmethod
    def _l2(A, B):
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def compute_F(self, X, V):
        n = self.n_balls
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(X, X), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * self.edges / l2_dist_power3
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[self.diag_mask]).min() > 1e-10)

            # here for minor precision issue with respect to the original script
            _X = X.T
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(_X[0, :],
                                       _X[0, :]).reshape(1, n, n),
                     np.subtract.outer(_X[1, :],
                                       _X[1, :]).reshape(1, n, n),
                     np.subtract.outer(_X[2, :],
                                       _X[2, :]).reshape(1, n, n)))).sum(axis=-1)
            F = F.T

            # adjust F scale
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

        return F

    def simulate_one_step(self):
        X, V = self.X, self.V
        F = self.compute_F(X, V)
        for obj in self.physical_objects:
            X, V = obj.update(X, V, F, self.delta_t)
        self.X, self.V = X, V
        return X, V

    def check(self):
        for obj in self.physical_objects:
            obj.check(self.X, self.V)

    def is_valid(self):
        if self.box_size:
            return np.all(self.X <= self.box_size) and np.all(self.X >= - self.box_size)
        else:
            return True  # no box size

    def configuration(self):
        cfg = {}
        for obj in self.physical_objects:
            _type = obj.type
            _node_idx = obj.node_idx
            if _type in cfg:
                cfg[_type].append(_node_idx)
            else:
                cfg[_type] = [_node_idx]
        return cfg

    def print(self):
        print('X:')
        print(self.X)
        print('V:')
        print(self.V)


def test():
    np.random.seed(10)
    system = System(n_isolated=10, n_stick=5, n_hinge=0)

    # np.random.seed(10)
    # system.X = np.random.rand(20, 3)
    # system.V = np.random.rand(20, 3)
    # charges = np.random.choice([1, -1], size=20).reshape(-1, 1)
    # system.edges = charges.dot(charges.transpose())
    # system.charges = charges
    # for obj in system.physical_objects:
    #     system.X, system.V = obj.initialize(system.X, system.V)

    system.print()
    steps = 5001
    ret = []
    for step in tqdm(range(steps)):
        system.simulate_one_step()
        ret.append((system.X.copy(), system.V.copy()))
        system.check()
    system.print()

    return ret


if __name__ == '__main__':
    ret = test()


