import numpy
from . import elim
from . import ortho


def calc_eig_vec_inv_pow(mat, eig, iters, eps=1.0e-15):
    a, mu = mat, eig
    n = mat.shape[0]
    a_mu_lu = elim.fact_lu(a - numpy.diag(mu * numpy.ones(n)))
    z = elim.solve_lower_unit(a_mu_lu, numpy.ones(n, dtype=a_mu_lu.dtype))
    for i in range(iters):
        z_old = z
        v = elim.solve_lu(a_mu_lu, z.copy())
        c = numpy.sum(z_old.conj() * v)
        c = c / numpy.abs(c) * numpy.linalg.norm(v)
        z = v / c
        if numpy.linalg.norm(z - z_old) < eps:
            break
    return z, i


def fact_upper_hess(mat, clean=True):
    a = mat
    n = a.shape[0]
    for i in range(n-2):
        v, beta = ortho.calc_house(a[i+1:, i])
        ortho.trans_house(v, beta, a[i+1:, i+1:])
        ortho.trans_house_trans(v, beta, a[:, i+1:])
        if clean:
            a[i+2:, i] = 0.0
    return a


def iter_qr_givens(mat):
    a = mat
    n = a.shape[0]
    c = numpy.zeros((n-1, 2))
    for i in range(n-1):
        c[i] = a[i:i+2, i]
        ortho.calc_givens(c[i])
        ortho.transform_givens(c[i], a[i:i+2, :])
    for i in range(n-1):
        ortho.transform_givens_trans(c[i], a[:, i:i+2])
    return a


def iter_qr_double_shift(mat):
    
    a = mat
    n = a.shape[0]
    
    s = a[-2, -2] + a[-1, -1]
    t = a[-2, -2] * a[-1, -1] - a[-2, -1] * a[-1, -2]
    
    x = numpy.zeros(3)
    
    x[0] = a[0, 0] * a[0, 0] + a[0, 1] * a[1, 0] - s * a[0, 0] + t
    x[1] = a[1, 0] * (a[0, 0] + a[1, 1] - s)
    x[2] = a[1, 0] * a[2, 1]
    
    for k in range(n-2):
        v, beta = ortho.calc_house(x)
        q, r = max(0, k-1), min(k+4, n)
        ortho.trans_house(v, beta, a[k:k+3, q:n])
        ortho.trans_house_trans(v, beta, a[:r, k:k+3])
        x[0] = a[k+1, k]
        x[1] = a[k+2, k]
        x[2] = a[r-1, k]
    
    v, beta = ortho.calc_house(x[:2])
    ortho.trans_house(v, beta, a[-2:, -3:])
    ortho.trans_house_trans(v, beta, a[:, -2:])

    return a


def check_vanish_sub_diag(mat, eps=1.0e-15):
    a = mat
    d_abs, s_abs = numpy.abs(a.diagonal()), numpy.abs(a.diagonal(-1))
    f = s_abs < (d_abs[:-1] + d_abs[1:]) * eps
    return f


def set_clean_sub_diag(mat, flag):
    a, f = mat, flag
    n = a.shape[0]
    a.flat[n::n+1][f] = 0.0
    return a


def check_conv(flag):
    f = flag
    return (f[:-1] | f[1:]).all()


def find_block(flag):
    f = flag
    n = f.shape[0] + 1
    g = f[:-1] | f[1:]
    s = numpy.searchsorted(numpy.cumsum(~g[::-1]), 1)
    g[-s:] = False
    t = numpy.searchsorted(numpy.cumsum(g[::-1]), 1)
    return n-t-2, n-s


def iter_qr_double_shift_sub(size, mat):
    
    n, a_full = size, mat
    m = a_full.shape[0]
    a = a_full[-n:, :n]
    
    s = a[-2, -2] + a[-1, -1]
    t = a[-2, -2] * a[-1, -1] - a[-2, -1] * a[-1, -2]
    
    x = numpy.zeros(3)
    
    x[0] = a[0, 0] * a[0, 0] + a[0, 1] * a[1, 0] - s * a[0, 0] + t
    x[1] = a[1, 0] * (a[0, 0] + a[1, 1] - s)
    x[2] = a[1, 0] * a[2, 1]
    
    for k in range(n-2):
        v, beta = ortho.calc_house(x)
        q, r = max(0, k-1), min(k+4, n)
        ortho.trans_house(v, beta, a_full[k+m-n:k+3+m-n, q:n])
        ortho.trans_house_trans(v, beta, a_full[m-n:r+m-n, k:k+3])
        x[0] = a[k+1, k]
        x[1] = a[k+2, k]
        x[2] = a[r-1, k]
    
    v, beta = ortho.calc_house(x[:2])
    ortho.trans_house(v, beta, a_full[-2:, n-3:n])
    ortho.trans_house_trans(v, beta, a_full[-n:, n-2:n])

    return a_full


def calc_eig_quasi_upper(mat, flag):
    a, f = mat, flag
    n = a.shape[0]
    e = numpy.zeros((n), dtype=numpy.complex128)
    r_flag = numpy.hstack([[True], f]) & numpy.hstack([f, [True]])
    r_sum = r_flag.sum()
    e[:r_sum] = a.diagonal()[r_flag]
    c_ind = numpy.arange(n-1)[~f]
    c_sum = c_ind.shape[0]
    c_mat = a[tuple(c_ind + numpy.indices((2, 2))[..., None])]
    s = c_mat[0, 0] + c_mat[1, 1]
    t = c_mat[0, 0] * c_mat[1, 1] - c_mat[0, 1] * c_mat[1, 0]
    c_det = s**2 - 4.0 * t
    c_sign = c_det >= 0.0
    e[r_sum:r_sum+c_sum][c_sign] = numpy.sqrt(c_det[c_sign])
    e[r_sum:r_sum+c_sum][~c_sign] = 1.0j * numpy.sqrt(-c_det[~c_sign])
    e[r_sum+c_sum:] = 0.5 * (s - e[r_sum:r_sum+c_sum])
    e[r_sum:r_sum+c_sum] = 0.5 * (s + e[r_sum:r_sum+c_sum])
    return e


def calc_vec_all(mat, eig, iters, eps=1.0e-15):
    a, es = mat, eig
    n, m = mat.shape[0], es.shape[0]
    v = numpy.zeros((m, n), dtype=es.dtype)
    for i, e in enumerate(es):
        v[:, i], _ = calc_eig_vec_inv_pow(a, e, iters, eps)
    return v


def calc_eig_match(eig1, eig2):
    e1, e2 = eig1, eig2
    p = numpy.argmin(numpy.abs(e1[:, None] - e2[None, :]), axis=1)
    return p


def calc_eig_error(eig1, eig2, perm):
    e1, e2, p = eig1, eig2, perm
    e = numpy.linalg.norm(e1 - e2[p], numpy.infty)
    return e


def calc_vec_error(vec1, vec2, perm):
    v1, v2, p = vec1, vec2, perm
    v2 = v2[:, p]
    c = numpy.sum(v1.conj() * v2, axis=0)
    c /= numpy.abs(c)
    v2 /= c[None, :]
    return numpy.linalg.norm(numpy.linalg.norm(v1 - v2, axis=0), numpy.infty)


def sum_eig_error(sol1, sol2):
    (e1, v1), (e2, v2) = sol1, sol2
    p = calc_eig_match(e1, e2)
    e_eig = calc_eig_error(e1, e2, p)
    e_vec = calc_vec_error(v1, v2, p)
    return e_eig, e_vec


def driver_qr_givens(mat, iters, eps=1.0e-15):
    a = mat
    fact_upper_hess(a)
    for i in range(iters):
        f = check_vanish_sub_diag(a, eps=eps)
        if check_conv(f):
            break
        iter_qr_givens(a)
    return a, f, i


# def driver_qr_double_shift(mat, iters, eps=1.0e-15):
#     a = mat
#     fact_upper_hess(a)
#     for i in range(iters):
#         f = check_vanish_sub_diag(a, eps=eps)
#         if check_conv(f):
#             break
#         iter_qr_double_shift(a)
#     return a, f, i


def driver_qr_impl(mat, iters, eps=1.0e-15):
    a = mat
    fact_upper_hess(a)
    for i in range(iters):
        f = check_vanish_sub_diag(a, eps=eps)
        if check_conv(f):
            break
        set_clean_sub_diag(a, f)
        s, t = find_block(f)
        iter_qr_double_shift_sub(t-s, a[:t, s:])
    return a, f, i
