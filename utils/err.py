import numpy
from . import lu


def est_norm_1(size, op, op_trans):
    n = size
    x = numpy.ones(n) / n
    while True:
        w = op(x)
        v = numpy.sign(w)
        z = op_trans(v)
        j = numpy.argmax(numpy.abs(z))
        if numpy.abs(z[j]) <= z.dot(x):
            nu = numpy.sum(numpy.abs(w))
            break
        x = numpy.zeros(n)
        x[j] = 1.0
    return nu


def est_cond_infty(size, mat):
    n, a = size, mat
    x = numpy.ones(n) / n
    nu_1 = numpy.max(numpy.sum(numpy.abs(a), axis=1))
    a_lu, p = lu.fact_lu_col(n, a.transpose().copy())
    q = numpy.argsort(p)
    nu_2 = est_norm_1(n, lambda b: lu.solve_lu(n, a_lu, b.copy()[p]), lambda b: lu.solve_lu_trans(n, a_lu, b)[q])
    kappa = nu_1 * nu_2
    return kappa


def est_error(size, mat, vec, sol):
    n, b, x, a = size, vec, sol, mat
    r = b - a.dot(x)
    gamma = numpy.max(numpy.abs(r))
    beta = numpy.max(numpy.abs(b))
    kappa = est_cond_infty(n, a)
    rho = kappa * gamma / beta
    return rho
