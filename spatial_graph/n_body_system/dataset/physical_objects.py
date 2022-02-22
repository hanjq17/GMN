import numpy as np

eps = 1e-6


def projection(va, vb):
    return np.dot(va, vb.T) / np.dot(vb, vb.T) * vb


def get_rotation_matrix(theta, d):
    x, y, z = d[0], d[1], d[2]
    M = np.zeros((3, 3))
    cos, sin = np.cos(theta), np.sin(theta)
    M[0][0] = cos + (1 - cos) * x * x
    M[0][1] = (1 - cos) * x * y - sin * z
    M[0][2] = (1 - cos) * x * z + sin * y
    M[1][0] = (1 - cos) * x * y + sin * z
    M[1][1] = cos + (1 - cos) * y * y
    M[1][2] = (1 - cos) * y * z - sin * x
    M[2][0] = (1 - cos) * x * z - sin * y
    M[2][1] = (1 - cos) * y * z + sin * x
    M[2][2] = cos + (1 - cos) * z * z
    return M


class PhysicalObject:
    def __init__(self, n_balls, node_idx, charge, type):
        self.n_balls, self.node_idx, self.type = n_balls, node_idx, type
        self.charge = charge
        assert len(node_idx) == n_balls == len(charge)

    def initialize(self, X, V):
        raise NotImplementedError()

    def update(self, X, V, F, delta_t):
        raise NotImplementedError()

    def check(self, X, V):
        raise NotImplementedError()


class Isolated(PhysicalObject):
    def __init__(self, n_balls, node_idx, charge, type):
        super().__init__(n_balls, node_idx, charge, type)

    def initialize(self, X, V):
        return X, V

    def update(self, X, V, F, delta_t):
        x, v, f = X[self.node_idx[0]], V[self.node_idx[0]], F[self.node_idx[0]]
        a = f / 1.

        v = v + a * delta_t
        x = x + v * delta_t
        X[self.node_idx[0]] = x
        V[self.node_idx[0]] = v
        return X, V

    def check(self, X, V):
        return True


class Stick(PhysicalObject):
    def __init__(self, n_balls, node_idx, charge, type):
        super().__init__(n_balls, node_idx, charge, type)
        self.xc, self.vc, self.wc = None, None, None
        self.length = None

    def initialize(self, X, V):
        # check and adjust the initial conditions
        x, v = X[self.node_idx], V[self.node_idx]
        x0, x1 = x[0], x[1]
        v0, v1 = v[0], v[1]
        m0, m1 = 1., 1.
        # the velocity along the stick should be the same for two nodes (0, 1)
        d = x1 - x0
        v0_pro, v1_pro = projection(v0, d), projection(v1, d)
        v0_vert, v1_vert = v0 - v0_pro, v1 - v1_pro
        average_v_pro = (v0_pro + v1_pro) / 2
        v0, v1 = v0_vert + average_v_pro, v1_vert + average_v_pro

        xc = (m0 * x0 + m1 * x1) / (m0 + m1)
        vc = (m0 * v0 + m1 * v1) / (m0 + m1)
        relative_v0, relative_v1 = v0 - vc, v1 - vc
        r0, r1 = x0 - xc, x1 - xc
        w0, w1 = np.cross(r0, relative_v0) / np.dot(r0, r0.T), np.cross(r1, relative_v1) / np.dot(r1, r1.T)
        assert np.sum(np.abs(w0 - w1)) < 1e-5
        # book-keeping
        self.xc, self.vc, self.wc = xc, vc, w0
        self.length = np.sqrt(np.sum(d ** 2))
        X[self.node_idx[0]], X[self.node_idx[1]] = x0, x1
        V[self.node_idx[0]], V[self.node_idx[1]] = v0, v1

        return X, V

    def update(self, X, V, F, delta_t):
        x, v, f = X[self.node_idx], V[self.node_idx], F[self.node_idx]
        x0, x1 = x[0], x[1]
        v0, v1 = v[0], v[1]
        f0, f1 = f[0], f[1]
        m0, m1 = 1., 1.
        xc, vc, wc = self.xc, self.vc, self.wc
        r0, r1 = x0 - xc, x1 - xc
        ac = (f0 + f1) / (m0 + m1)

        # update vc, xc
        vc = vc + ac * delta_t
        xc = xc + vc * delta_t

        # update wc
        J = m0 * np.dot(r0, r0.T) + m1 * np.dot(r1, r1.T)
        M = np.cross(r0, f0) + np.cross(r1, f1)
        beta = M / J

        wc = wc + beta * delta_t

        wc_norm = np.sqrt(np.dot(wc, wc.T))
        theta = wc_norm * delta_t

        M = get_rotation_matrix(theta, wc / wc_norm)
        _r0 = np.matmul(M, r0.T).T
        _r1 = np.matmul(M, r1.T).T

        # update x and v
        x0, x1 = xc + _r0, xc + _r1
        v0, v1 = vc + np.cross(wc, _r0), vc + np.cross(wc, _r1)  # here, we use the updated r (instead of original r)

        # book-keeping
        self.xc, self.vc, self.wc = xc, vc, wc
        X[self.node_idx[0]], X[self.node_idx[1]] = x0, x1
        V[self.node_idx[0]], V[self.node_idx[1]] = v0, v1

        return X, V

    def check(self, X, V):
        x, v = X[self.node_idx], V[self.node_idx]
        x0, x1 = x[0], x[1]
        v0, v1 = v[0], v[1]

        d = x1 - x0
        v0_pro, v1_pro = projection(v0, d), projection(v1, d)

        assert np.sum(np.abs(v0_pro - v1_pro)) < eps
        length = np.sqrt(np.sum(d ** 2))
        assert np.abs(length - self.length) < eps


class Hinge(PhysicalObject):
    def __init__(self, n_balls, node_idx, charge, type):
        super().__init__(n_balls, node_idx, charge, type)
        self.w1, self.w2 = None, None
        self.length1, self.length2 = None, None

    def initialize(self, X, V):
        # check and adjust the initial conditions
        x, v = X[self.node_idx], V[self.node_idx]
        x0, x1, x2 = x[0], x[1], x[2]
        v0, v1, v2 = v[0], v[1], v[2]
        # the velocity along the two beams should be the same for nodes (0, 1) and (0, 2), respectively
        d1, d2 = x1 - x0, x2 - x0
        v0_pro1, v0_pro2 = projection(v0, d1), projection(v0, d2)
        v1_pro, v2_pro = projection(v1, d1), projection(v2, d2)
        v1_vert, v2_vert = v1 - v1_pro, v2 - v2_pro
        v1, v2 = v0_pro1 + v1_vert, v0_pro2 + v2_vert

        r1, r2 = x1 - x0, x2 - x0
        v01, v02 = v1 - v0, v2 - v0
        w1, w2 = np.cross(r1, v01) / np.dot(r1, r1.T), np.cross(r2, v02) / np.dot(r2, r2.T)

        # book-keeping
        self.w1, self.w2 = w1, w2
        X[self.node_idx[0]], X[self.node_idx[1]], X[self.node_idx[2]] = x0, x1, x2
        V[self.node_idx[0]], V[self.node_idx[1]], V[self.node_idx[2]] = v0, v1, v2

        self.length1, self.length2 = np.sqrt(np.sum(d1 ** 2)), np.sqrt(np.sum(d2 ** 2))

        return X, V

    def update(self, X, V, F, delta_t):
        x, v, f = X[self.node_idx], V[self.node_idx], F[self.node_idx]
        x0, x1, x2 = x[0], x[1], x[2]
        v0, v1, v2 = v[0], v[1], v[2]
        f0, f1, f2 = f[0], f[1], f[2]

        _f = f0 + f1 + f2
        r01, r02 = x1 - x0, x2 - x0
        v01, v02 = v1 - v0, v2 - v0
        w1, w2 = self.w1, self.w2
        e01, e02 = r01 / np.sqrt(np.dot(r01, r01.T)), r02 / np.sqrt(np.dot(r02, r02.T))
        e01, e02 = e01.reshape(1, -1), e02.reshape(1, -1)
        A = np.eye(3) + np.matmul(e01.T, e01) + np.matmul(e02.T, e02)
        a = _f / 1. - np.cross(w1, v01) - np.cross(w2, v02)
        a = a - np.matmul((np.eye(3) - np.matmul(e01.T, e01)), f1 / 1.) - np.matmul((np.eye(3) - np.matmul(e02.T, e02)),
                                                                                    f2 / 1.)
        a0 = np.matmul(np.linalg.inv(A), a)

        # update x0, v0
        v0 = v0 + a0 * delta_t
        x0 = x0 + v0 * delta_t

        # update w1, w2
        beta1 = np.cross(r01, f1 - 1. * a0) / (1. * np.dot(r01, r01.T))
        beta2 = np.cross(r02, f2 - 1. * a0) / (1. * np.dot(r02, r02.T))
        w1 = w1 + beta1 * delta_t
        w2 = w2 + beta2 * delta_t

        # update x, v
        w1_norm = np.sqrt(np.dot(w1, w1.T))
        theta = w1_norm * delta_t
        M = get_rotation_matrix(theta, w1 / w1_norm)
        _r01 = np.matmul(M, r01.T).T
        x1 = x0 + _r01

        w2_norm = np.sqrt(np.dot(w2, w2.T))
        theta = w2_norm * delta_t
        M = get_rotation_matrix(theta, w2 / w2_norm)
        _r02 = np.matmul(M, r02.T).T
        x2 = x0 + _r02

        v1, v2 = v0 + np.cross(w1, _r01), v0 + np.cross(w2, _r02)

        # book-keeping
        self.w1, self.w2 = w1, w2
        X[self.node_idx[0]], X[self.node_idx[1]], X[self.node_idx[2]] = x0, x1, x2
        V[self.node_idx[0]], V[self.node_idx[1]], V[self.node_idx[2]] = v0, v1, v2

        return X, V

    def check(self, X, V):
        x, v  = X[self.node_idx], V[self.node_idx]
        x0, x1, x2 = x[0], x[1], x[2]
        v0, v1, v2 = v[0], v[1], v[2]
        d1 = x1 - x0
        d2 = x2 - x0

        length1, length2 = np.sqrt(np.sum(d1 ** 2)), np.sqrt(np.sum(d2 ** 2))
        assert np.abs(length1 - self.length1) < eps
        assert np.abs(length2 - self.length2) < eps

        v0_pro1, v0_pro2 = projection(v0, d1), projection(v0, d2)
        v1_pro, v2_pro = projection(v1, d1), projection(v2, d2)
        assert np.sum(np.abs(v0_pro1 - v1_pro)) < eps
        assert np.sum(np.abs(v0_pro2 - v2_pro)) < eps










