import numpy
from . import elim


def calc_house(vec):
    x = vec
    eta = numpy.max(numpy.abs(x))
    y = x / eta
    sigma = numpy.sum(y[1:]**2)
    alpha = numpy.sqrt(sigma + y[0]**2)
    x[0] = alpha * eta
    if y[0] <= 0:
        gamma = y[0] - alpha
    else:
        gamma = -sigma / (y[0] + alpha)
    if sigma == 0.0 and y[0] >= 0.0:
        beta = 0.0
        x[1:] = 0.0
    else:
        beta = 2.0 * gamma**2 / (sigma + gamma**2)
        x[1:] = y[1:] / gamma
    return x, beta


def trans_house(vec, beta, mat, lead=True):
    x, a = vec, mat
    if lead:
        x = x[1:]
    d = (x.dot(a[1:, :]) + a[0, :]) * beta
    a[0, :] -= d
    a[1:, :] -= x[:, None] * d[None, :]
    return a


def trans_house_trans(vec, beta, mat, lead=True):
    x, a = vec, mat
    if lead:
        x = x[1:]
    d = (a[:, 1:].dot(x) + a[:, 0]) * beta
    a[:, 0] -= d
    a[:, 1:] -= x[None, :] * d[:, None]
    return a


def fact_qr_house(mat):
    a = mat
    n = a.shape[1]
    beta = numpy.zeros(n)
    for i in range(n):
        _, beta[i] = calc_house(a[i:, i])
        d = (a[i+1:, i].dot(a[i+1:, i+1:]) + a[i, i+1:]) * beta[i]
        a[i, i+1:] -= d
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * d[None, :]
    return a, beta


def solve_ortho_house(mat, beta, vec):
    a, b = mat, vec
    n = mat.shape[0]
    for i in range(n):
        trans_house(a[i:, i], beta[i], b[i:, None])
    return b


def solve_qr(fact, beta, vec):
    qr, b = fact, vec
    a = fact
    n = qr.shape[0]
#     for i in range(n):
#         d = (b[i] + b[i+1:].dot(a[i+1:, i])) * beta[i]                        
#         b[i] -= d
#         b[i+1:] -= d * a[i+1:, i]
#     for i in range(n-1, -1, -1):
#         b[i] /= a[i, i]
#         b[:i] -= a[:i, i] * b[i]
    solve_ortho_house(qr, beta, b)
    elim.solve_upper(qr, b)
    return b


def calc_givens(val):
    x = val
    r = numpy.linalg.norm(val, 2.0)
    x /= r
    return x


def transform_givens(coef, mat):
    c, a = coef, mat
    m = numpy.array([[c[0], c[1]], [-c[1], c[0]]])
    a[:, :] = m.dot(a)
    return a


def transform_givens_trans(coef, mat):
    c, a = coef, mat
    m = numpy.array([[c[0], -c[1]], [c[1], c[0]]])
    a[:, :] = a.dot(m)
    return a
