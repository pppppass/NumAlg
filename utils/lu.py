import numpy


def solve_diag(size, mat, vec):
    n, b, d = size, vec, mat
    b /= d.diagonal()
    return b


def solve_upper(size, mat, vec):
    n, b, u = size, vec, mat
    for i in range(n-1, -1, -1):
        b[i] /= u[i, i]
        b[:i] -= b[i] * u[:i, i]
    return b


def solve_lower(size, mat, vec):
    n, b, l = size, vec, mat
    for i in range(n):
        b[i] /= l[i, i]
        b[i+1:] -= b[i] * l[i+1:, i]
    return b


def solve_upper_unit(size, mat, vec):
    n, b, u = size, vec, mat
    for i in range(n-1, 0, -1):
        b[:i] -= b[i] * u[:i, i]
    return b


def solve_lower_unit(size, mat, vec):
    n, b, l = size, vec, mat
    for i in range(n-1):
        b[i+1:] -= b[i] * l[i+1:, i]
    return b


def fact_lu(size, mat):
    n, a = size, mat
    for i in range(n-1):
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a


def solve_lu(size, fact, vec):
    n, b, lu = size, vec, fact
    solve_lower_unit(n, lu, b)
    solve_upper(n, lu, b)
    return b


def solve_lu_trans(size, fact, vec):
    n, b, lu = size, vec, fact
    solve_lower(n, lu.transpose(), b)
    solve_upper_unit(n, lu.transpose(), b)
    return b


def fact_lu_col(size, mat):
    n, a = size, mat
    a = mat
    p = numpy.arange(n)
    for i in range(n-1):
        j = numpy.argmax(numpy.abs(a[i:, i])) + i
        a[[i, j], :] = a[[j, i], :]
        p[[i, j]] = p[[j, i]]
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a, p


def solve_lu_col(size, mat, vec, perm):
    n, b, lu, p = size, vec, mat, perm
    b = b[p]
    solve_lu(n, lu, b)
    return b


def fact_lu_full(size, mat):
    n, a = size, mat
    p, q = numpy.arange(n), numpy.arange(n)
    for i in range(n-1):
        j, k = numpy.unravel_index(numpy.argmax(a[i:, i:]), (n-i, n-i))
        j, k = j+i, k+i
        a[[i, j], :] = a[[j, i], :]
        a[:, [i, k]] = a[:, [k, i]]
        p[[i, j]] = p[[j, i]]
        q[[i, k]] = q[[k, i]]
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a, p, q


def solve_lu_full(size, mat, vec, perm_x, perm_y):
    n, b, lu, p, q = size, vec, mat, perm_x, perm_y
    b = b[p]
    solve_lu(n, lu, b)
    b = b[numpy.argsort(q)]
    return b


def fact_chol(size, mat):
    n, a = size, mat
    for i in range(n):
        a[i:, i] -= a[i:, :i].dot(a[i, :i])
        a[i, i] = numpy.sqrt(a[i, i])
        a[i+1:, i] /= a[i, i]
    return a


def solve_chol(size, mat, vec):
    n, b, llt = size, vec, mat
    solve_lower(n, llt, b)
    solve_upper(n, llt.transpose(), b)
    return b


def fact_ldl(size, mat):
    n, a = size, mat
    v = numpy.zeros(n)
    d = a.diagonal()
    for i in range(n):
        v[:i] = d[:i] * a[i, :i]
        a[i:, i] -= a[i:, :i].dot(v[:i])
        a[i+1:, i] /= a[i, i]
    return a


def solve_ldl(size, mat, vec):
    n, b, ldlt = size, vec, mat
    solve_lower_unit(n, ldlt, b)
    solve_diag(n, ldlt, b)
    solve_upper_unit(n, ldlt.transpose(), b)
    return b


def fact_lu_band(size, mat, widths):
    n, kl, kh, a = size, *widths, mat
    for i in range(n-1):
        a[i+1:i+kl+1, i] /= a[i, i]
        a[i+1:i+kl+1, i+1:i+kh+1] -= a[i+1:i+kl+1, i:i+1] * a[i:i+1, i+1:i+kh+1]
    return a


def solve_upper_band(size, mat, vec, width):
    n, k, b, u = size, width, vec, mat
    for i in range(n-1, 0, -1):
        b[i] /= u[i, i]
        b[i-k-n:i-n] -= b[i] * u[i-k-n:i-n, i]
    b[0] /= u[0, 0]
    return b


def solve_lower_unit_band(size, mat, vec, width):
    n, k, b, l = size, width, vec, mat
    for i in range(n-1):
        b[i+1:i+k+1] -= b[i] * l[i+1:i+k+1, i]
    return b


def solve_lu_band(size, mat, vec, widths):
    n, kl, kh, b, lu = size, *widths, vec, mat
    solve_lower_unit_band(n, lu, b, kl)
    solve_upper_band(n, lu, b, kh)
    return b
