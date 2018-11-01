import numpy


def calc_house(size, vec):
    n, x = size, vec
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


def fact_qr_house(size, mat):
    n, a = size, mat
    beta = numpy.zeros(n)
    for i in range(n):
        _, beta[i] = calc_house(n-i, a[i:, i])
        d = (a[i+1:, i].dot(a[i+1:, i+1:]) + a[i, i+1:]) * beta[i]
        a[i, i+1:] -= d
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * d[None, :]
    return a, beta


def solve_qr(size, mat, coef, vec):
    n, b, beta, a = size, vec, coef, mat
    for i in range(n):
        d = (b[i] + b[i+1:].dot(a[i+1:, i])) * beta[i]
        b[i] -= d
        b[i+1:] -= d * a[i+1:, i]
    for i in range(n-1, -1, -1):
        b[i] /= a[i, i]
        b[:i] -= a[:i, i] * b[i]
    return b
