import numpy
import scipy.sparse


def get_lap_1d(size):
    n = size
    a = numpy.zeros((n, n))
    a.flat[::n+1] = 2.0
    a.flat[1::n+1] = -1.0
    a.flat[n::n+1] = -1.0
    return a

def get_lap_2d_sparse(size):
    n = size
    d = numpy.zeros((5, n, n))
    d[0, :, :] = 4.0
    d[1, :, :] = -1.0
    d[1, :, 0] = 0.0
    d[2, :, :] = -1.0
    d[3, :, :] = -1.0
    d[3, :, -1] = 0.0
    d[4, :, :] = -1.0
    a = scipy.sparse.dia_matrix((d.reshape(5, -1), [0, 1, n, -1, -n]), (n*n, n*n)).tocsr()
    return a


def get_lap_2d(size):
    n = size
    a_sp = get_lap_2d_sparse(n)
    a = a_sp.todense().A
    return a


def get_friend(coef):
    n = coef.shape[0]
    a = numpy.zeros((n, n))
    a.flat[n::n+1] = 1.0
    a[::-1, -1] = -coef
    return a
