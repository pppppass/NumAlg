import numpy
import scipy.sparse


def cvrt_list_to_csr(size, list_):
    data = []
    rows = []
    cols = []
    for v, x1, x2, y1, y2 in list_:
        if 0 <= y1 < size and 0 <= y2 < size:
            data.append(v)
            rows.append(x1*size + x2)
            cols.append(y1*size + y2)
    mat = scipy.sparse.coo_matrix(
        (data, (rows, cols)), shape=(size**2, size**2))
    mat = mat.tocsr()
    return mat


def fact_lu(mat):
    n = mat.shape[0]
    a = mat
    for i in range(n-1):
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a


def solve_upper(mat, vec):
    n = mat.shape[0]
    u, b = mat, vec
    for i in range(n-1, -1, -1):
        b[i] /= u[i, i]
        b[:i] -= b[i] * u[:i, i]
    return b


def solve_lower_unit(mat, vec):
    n = mat.shape[0]
    l, b = mat, vec
    for i in range(n-1):
        b[i+1:] -= b[i] * l[i+1:, i]
    return b


def solve_lu_fact(mat, vec):
    lu, b = mat, vec
    solve_lower_unit(lu, b)
    solve_upper(lu, b)
    return b


def solve_lu(mat, vec):
    fact_lu(mat)
    solve_lu_fact(mat, vec)
    return vec


def calc_safe_sqrt(var):
    return numpy.sqrt(numpy.clip(var, 0.0, numpy.infty))


def fact_chol(mat):
    n = mat.shape[0]
    a = mat
    for i in range(n):
        a[i, i] = calc_safe_sqrt(a[i, i])
        a[i+1:, i] /= a[i, i]
        for j in range(i+1, n):
            a[j:, j] -= a[j:, i] * a[j, i]
    return a


def solve_lower(mat, vec):
    n = mat.shape[0]
    l, b = mat, vec
    for i in range(n):
        b[i] /= l[i, i]
        b[i+1:] -= b[i] * l[i+1:, i]
    return b


def solve_upper_trans(mat, vec):
    n = mat.shape[0]
    l, b = mat, vec
    for i in range(n-1, -1, -1):
        b[i] /= l[i, i]
        b[:i] -= b[i] * l[i, :i]
    return b


def solve_chol_fact(mat, vec):
    ll, b = mat, vec
    solve_lower(ll, b)
    solve_upper_trans(ll, b)
    return b


def solve_chol(mat, vec):
    fact_chol(mat)
    solve_chol_fact(mat, vec)
    return vec


def fact_ldl(mat):
    n = mat.shape[0]
    a = mat
    t = numpy.zeros(n)
    for i in range(n-1):
        t[i+1:] = a[i+1:, i]
        a[i+1:, i] /= a[i, i]
        for j in range(i+1, n):
            a[j:, j] -= a[j:, i] * t[j]
    return a


def solve_upper_unit_trans(mat, vec):
    n = mat.shape[0]
    l, b = mat, vec
    for i in range(n-1, 0, -1):
        b[:i] -= b[i] * l[i, :i]
    return b


def solve_diag(mat, vec):
    d, b = mat, vec
    b /= d.diagonal()
    return b


def solve_ldl_fact(mat, vec):
    ldl, b = mat, vec
    solve_lower_unit(ldl, b)
    solve_diag(ldl, b)
    solve_upper_unit_trans(ldl, b)
    return b


def solve_ldl(mat, vec):
    fact_ldl(mat)
    solve_ldl_fact(mat, vec)
    return vec


def fact_lu_band(mat, width):
    n, kl, kh = mat.shape[0], *width
    a = mat
    for i in range(n-1):
        a[i+1:i+kl+1, i] /= a[i, i]
        a[i+1:i+kl+1, i+1:i+kh+1] -= a[i+1:i+kl+1, i:i+1] * a[i:i+1, i+1:i+kh+1]
    return a


def solve_upper_band(mat, vec, width):
    n, k = mat.shape[0], width
    u, b = mat, vec
    for i in range(n-1, 0, -1):
        b[i] /= u[i, i]
        b[i-k-n:i-n] -= b[i] * u[i-k-n:i-n, i]
    b[0] /= u[0, 0]
    return b


def solve_lower_unit_band(mat, vec, width):
    n, k = mat.shape[0], width
    l, b = mat, vec
    for i in range(n-1):
        b[i+1:i+k+1] -= b[i] * l[i+1:i+k+1, i]
    return b


def solve_lu_band_fact(mat, vec, width):
    kl, kh = width
    lu, b = mat, vec
    solve_lower_unit_band(lu, b, kl)
    solve_upper_band(lu, b, kh)
    return b


def solve_lu_band(mat, vec, width):
    fact_lu_band(mat, width)
    solve_lu_band_fact(mat, vec, width)
    return vec
