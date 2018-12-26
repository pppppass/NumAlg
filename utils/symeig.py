import numpy
from . import ortho


def calc_jacobi_givens(mat):
    a = mat
    t = (a[1, 1] - a[0, 0]) / 2.0 / a[0, 1]
    t = numpy.sign(t + 1.0e-15) / (numpy.abs(t) + numpy.sqrt(1.0 + t**2))
    c = 1.0 / numpy.sqrt(1.0 + t**2)
    s = t * c
    v = numpy.array([c, -s])
    return v


def trans_jacobi_givens(mat, trans, pos):
    a, o, (p, q) = mat, trans, pos
    v = calc_jacobi_givens(a[[[p], [q]], [[p, q]]])
    a[[p, q], :] = ortho.transform_givens(v, a[[p, q], :])
    a[:, [p, q]] = ortho.transform_givens_trans(v, a[:, [p, q]])
    o[[p, q], :] = ortho.transform_givens(v, o[[p, q], :])
    return a, o


def iter_jacobi_rec(mat, trans, delta):
    a, o = mat, trans
    n = a.shape[0]
    f = False
    for i in range(n):
        for j in range(i+1, n):
            if numpy.abs(a[i, j]) > delta:
                trans_jacobi_givens(a, o, (i, j))
                f = True
    return a, o, f


def driver_thre_jacobi(mat, delta, gamma, max_iter, eps=1.0e-15):
    a = mat
    n = a.shape[0]
    o = numpy.eye(n)
    for i in range(max_iter):
        _, _, f = iter_jacobi_rec(a, o, delta)
        if not f:
            if delta < eps:
                break
            delta /= gamma
    return a.diagonal(), o.transpose(), i
