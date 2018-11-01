import numpy


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

def fact_lu(size, mat):
    n = size
    a = mat
    for i in range(n):
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i:i+1, i+1:] * a[i+1:, i:i+1]
    return a

def fact_lu_col(size, mat):
    n = size
    a = mat
    p = numpy.arange(n)
    for i in range(n):
        j = numpy.argmax(a[i:, i])
        a[[i, i+j], :] = a[[i+j, i], :]
        p[[i, i+j]] = p[[i, i+j]]
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i:i+1, i+1:] * a[i+1:, i:i+1]
    return a, p

def solve_lu(size, fact, vec):
    n = size
    b = vec
    lu = fact
    for i in range(n):
        b[i+1:] -= b[i] * lu[i+1:, i]
    for i in range(n-1, -1, -1):
        b[i] /= lu[i, i]
        b[:i] -= b[i] * lu[:i, i]
    return b

def solve_lu_trans(size, fact, vec):
    n = size
    b = vec
    lu = fact
    for i in range(n):
        b[i] /= lu[i, i]
        b[i+1:] -= b[i] * lu[i, i+1:]
    for i in range(n-1, -1, -1):
        b[:i] -= b[i] * lu[i, :i]
    return b

def est_cond_infty(size, mat):
    n = size
    a = mat
    x = numpy.ones(n) / n
    nu_1 = numpy.max(numpy.sum(numpy.abs(a), axis=1))
    lu, p = fact_lu_col(n, a.transpose().copy())
    nu_2 = est_norm_1(n, lambda b: solve_lu(n, lu, b.copy()[p]), lambda b: solve_lu_trans(n, lu, b[p]))
    kappa = nu_1 * nu_2
    return kappa

