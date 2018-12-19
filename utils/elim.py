import numpy


def solve_diag(mat, vec):
    d, b = mat, vec
    b /= d.diagonal()
    return b


def solve_upper(mat, vec):
    u, b = mat, vec
    n = u.shape[0]
    for i in range(n-1, -1, -1):
        b[i] /= u[i, i]
        b[:i] -= b[i] * u[:i, i]
    return b


def solve_lower(mat, vec):
    l, b = mat, vec
    n = l.shape[0]
    for i in range(n):
        b[i] /= l[i, i]
        b[i+1:] -= b[i] * l[i+1:, i]
    return b


def solve_upper_unit(mat, vec):
    u, b = mat, vec
    n = u.shape[0]
    for i in range(n-1, 0, -1):
        b[:i] -= b[i] * u[:i, i]
    return b


def solve_lower_unit(mat, vec):
    l, b = mat, vec
    n = l.shape[0]
    for i in range(n-1):
        b[i+1:] -= b[i] * l[i+1:, i]
    return b


def fact_lu(mat):
    a = mat
    n = a.shape[0]
    for i in range(n-1):
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a


def solve_lu(fact, vec):
    lu, b = fact, vec
    solve_lower_unit(lu, b)
    solve_upper(lu, b)
    return b


def solve_lu_trans(fact, vec):
    lu, b = fact, vec
    solve_lower(lu.transpose(), b)
    solve_upper_unit(lu.transpose(), b)
    return b


def fact_lu_col(mat):
    a = mat
    n = a.shape[0]
    p = numpy.arange(n)
    for i in range(n-1):
        j = numpy.argmax(numpy.abs(a[i:, i])) + i
        a[[i, j], :] = a[[j, i], :]
        p[[i, j]] = p[[j, i]]
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a, p


def solve_lu_col(fact, vec, perm):
    lu, b, p = fact, vec, perm
    b = b[p]
    solve_lu(lu, b)
    return b


def fact_lu_full(mat):
    a = mat
    n = a.shape[0]
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


def solve_lu_full(fact, vec, perm_row, perm_col):
    lu, b, p, q = fact, vec, perm_row, perm_col
    b = b[p]
    solve_lu(lu, b)
    b = b[numpy.argsort(q)]
    return b


def fact_chol(mat):
    a = mat
    n = a.shape[0]
    for i in range(n):
        a[i:, i] -= a[i:, :i].dot(a[i, :i])
        a[i, i] = numpy.sqrt(a[i, i])
        a[i+1:, i] /= a[i, i]
    return a


def solve_chol(fact, vec):
    llt, b = fact, vec
    solve_lower(llt, b)
    solve_upper(llt.transpose(), b)
    return b


def fact_ldl(mat):
    a = mat
    n = a.shape[0]
    v = numpy.zeros(n)
    d = a.diagonal()
    for i in range(n):
        v[:i] = d[:i] * a[i, :i]
        a[i:, i] -= a[i:, :i].dot(v[:i])
        a[i+1:, i] /= a[i, i]
    return a


def solve_ldl(fact, vec):
    ldlt, b = fact, vec
    solve_lower_unit(ldlt, b)
    solve_diag(ldlt, b)
    solve_upper_unit(ldlt.transpose(), b)
    return b


def fact_lu_band(mat, widths):
    a, kl, kh = mat, *widths
    n = a.shape[0]
    for i in range(n-1):
        a[i+1:i+kl+1, i] /= a[i, i]
        a[i+1:i+kl+1, i+1:i+kh+1] -= a[i+1:i+kl+1, i:i+1] * a[i:i+1, i+1:i+kh+1]
    return a


def solve_upper_band(mat, vec, width):
    u, b, k = mat, vec, width
    n = u.shape[0]
    for i in range(n-1, 0, -1):
        b[i] /= u[i, i]
        b[i-k-n:i-n] -= b[i] * u[i-k-n:i-n, i]
    b[0] /= u[0, 0]
    return b


def solve_lower_unit_band(mat, vec, width):
    l, b, k = mat, vec, width
    n = l.shape[0]
    for i in range(n-1):
        b[i+1:i+k+1] -= b[i] * l[i+1:i+k+1, i]
    return b


def solve_lu_band(fact, vec, widths):
    lu, b, kl, kh = fact, vec, *widths
    solve_lower_unit_band(lu, b, kl)
    solve_upper_band(lu, b, kh)
    return b
