import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.fftpack


def calc_a_x(size, u):
    n = size
    a_x_u = numpy.zeros((n-1, n))
    a_x_u[:, 1:-1] = 4.0 * u[:, 1:-1]
    a_x_u[:, [0, -1]] = 3.0 * u[:, [0, -1]]
    a_x_u[:, :-1] -= u[:, 1:]
    a_x_u[:, 1:] -= u[:, :-1]
    a_x_u[:-1, :] -= u[1:, :]
    a_x_u[1:, :] -= u[:-1, :]
    return a_x_u


def calc_a_y(size, v):
    n = size
    a_y_v = numpy.zeros((n, n-1))
    a_y_v[1:-1, :] = 4.0 * v[1:-1, :]
    a_y_v[[0, -1], :] = 3.0 * v[[0, -1], :]
    a_y_v[:-1, :] -= v[1:, :]
    a_y_v[1:, :] -= v[:-1, :]
    a_y_v[:, :-1] -= v[:, 1:]
    a_y_v[:, 1:] -= v[:, :-1]
    return a_y_v


def calc_b_x(size, p):
    n = size
    h = 1.0 / n
    b_x_p = h * (p[1:, :] - p[:-1, :])
    return b_x_p


def calc_b_y(size, p):
    n = size
    h = 1.0 / n
    b_y_p = h * (p[:, 1:] - p[:, :-1])
    return b_y_p


def calc_b_x_t(size, u):
    n = size
    h = 1.0 / n
    b_x_t_u = numpy.zeros((n, n))
    b_x_t_u[:-1, :] -= h * u
    b_x_t_u[1:, :] += h * u
    return b_x_t_u


def calc_b_y_t(size, v):
    n = size
    h = 1.0 / n
    b_y_t_v = numpy.zeros((n, n))
    b_y_t_v[:, :-1] -= h * v
    b_y_t_v[:, 1:] += h * v
    return b_y_t_v


def get_mat(size):
    
    n = size
    h = 1.0/n
    
    a_x_data = numpy.zeros((5, n-1, n))
    a_x_data[0, :, :] = 4.0
    a_x_data[0, :, [0, -1]] = 3.0
    a_x_data[1, :, :] = -1.0
    a_x_data[1, :, 0] = 0.0
    a_x_data[2, :, :] = -1.0
    a_x_data[2, 0, :] = 0.0
    a_x_data[3, :, :] = -1.0
    a_x_data[3, :, -1] = 0.0
    a_x_data[4, :, :] = -1.0
    a_x_data[4, -1, :] = 0.0
    a_x = scipy.sparse.dia_matrix((a_x_data.reshape(5, -1), [0, 1, n, -1, -n]), shape=((n-1)*n, (n-1)*n)).tocoo()
    i_full_half = numpy.argsort(numpy.arange((n-1)*n).reshape(n-1, n).transpose().flat)
    a_y = a_x.copy()
    a_y.row, a_y.col = i_full_half[a_y.row], i_full_half[a_y.col]
    
    b_x_data = numpy.zeros((2, n, n))
    b_x_data[0, :, :] = -1.0 * h
    b_x_data[0, -1, :] = 0.0
    b_x_data[1, :, :] = 1.0 * h
    b_x_data[1, 0, :] = 0.0
    b_x = scipy.sparse.dia_matrix((b_x_data.reshape(2, -1), [0, n]), shape=((n-1)*n, n*n)).tocoo()
    i_full_full = numpy.argsort(numpy.arange(n*n).reshape(n, n).transpose().flat)
    b_y = b_x.copy()
    b_y.row, b_y.col = i_full_half[b_y.row], i_full_full[b_y.col]
    
    a = scipy.sparse.vstack([scipy.sparse.hstack([m for m in ms]) for ms in [
        [a_x, scipy.sparse.coo_matrix(((n-1)*n, n*(n-1))), b_x],
        [scipy.sparse.coo_matrix((n*(n-1), (n-1)*n)), a_y, b_y],
        [b_x.transpose(), b_y.transpose(), scipy.sparse.coo_matrix((n*n, n*n))]
    ]]).tocsr()
    
    return a


def sol_dir(size, mat, c_x, c_y, c_i, solve_func=lambda a, b: scipy.sparse.linalg.spsolve(a, b)):
    n, a = size, mat
    b = numpy.hstack([c_x.flat, c_y.flat, c_i.flat])
    if isinstance(a, scipy.sparse.spmatrix):
        x = scipy.sparse.linalg.spsolve(a, b)
    elif isinstance(a, numpy.ndarray):
        x = numpy.linalg.solve(a, b)
    else:
        x = a(b)
    u_sol = x[:(n-1)*n].reshape((n-1, n))
    v_sol = x[(n-1)*n:(n-1)*n+n*(n-1)].reshape((n, n-1))
    p_sol = x[(n-1)*n+n*(n-1):].reshape((n, n))
    return u_sol, v_sol, p_sol
