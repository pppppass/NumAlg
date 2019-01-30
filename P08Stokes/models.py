import numpy
import mat
import op


def get_models_1(size):
    
    n = size
    h = 1.0 / n
    
    t = numpy.linspace(0.0, 1.0, 2*n+1)
    x_full, x_half = t[2:-1:2, None], t[1::2, None]
    y_full, y_half = t[None, 2:-1:2], t[None, 1::2]

    u_ana = (1.0 - numpy.cos(2.0 * numpy.pi * x_full)) * numpy.sin(2.0 * numpy.pi * y_half)
    v_ana = -(1.0 - numpy.cos(2.0 * numpy.pi * y_full)) * numpy.sin(2.0 * numpy.pi * x_half)
    p_ana = numpy.broadcast_arrays(x_half**3 / 3.0 - 1.0 / 12.0, y_half)[0]
    
    f_x = -4.0 * numpy.pi**2 * (2.0 * numpy.cos(2.0 * numpy.pi * x_full) - 1.0) * numpy.sin(2.0 * numpy.pi * y_half) + x_full**2
    f_y = 4.0 * numpy.pi**2 * (2.0 * numpy.cos(2.0 * numpy.pi * y_full) - 1.0) * numpy.sin(2.0 * numpy.pi * x_half)
    
    c_w = 2.0 * numpy.pi * (1.0 - numpy.cos(2.0 * numpy.pi * y_full))
    c_s = -2.0 * numpy.pi * (1.0 - numpy.cos(2.0 * numpy.pi * x_full))
    c_e = -2.0 * numpy.pi * (1.0 - numpy.cos(2.0 * numpy.pi * y_full))
    c_n = 2.0 * numpy.pi * (1.0 - numpy.cos(2.0 * numpy.pi * x_full))
    d_w = numpy.zeros_like(y_half)
    d_s = numpy.zeros_like(x_half)
    d_e = numpy.zeros_like(y_half)
    d_n = numpy.zeros_like(x_half)
    
    return u_ana, v_ana, p_ana, (f_x, f_y), (c_w, c_s, c_e, c_n), (d_w, d_s, d_e, d_n)


def get_models_2(size, kappa=10.0):
    
    n = size
    h = 1.0 / n
    
    t = numpy.linspace(0.0, 1.0, 2*n+1)
    x_full, x_half = t[2:-1:2, None], t[1::2, None]
    y_full, y_half = t[None, 2:-1:2], t[None, 1::2]

    u_ana = numpy.sin(kappa * (x_full**2 - y_half))
    v_ana = 2.0 * x_half * numpy.sin(kappa * (x_half**2 - y_full))
    p_ana = 2.0 * kappa * x_half * numpy.cos(kappa * (x_half**2 - y_half))
    
    f_x = kappa**2 * numpy.sin(kappa * (x_full**2 - y_half))
    f_y = (
          4.0 * kappa**2 * (2.0 * x_half**3 + x_half) * numpy.sin(kappa * (x_half**2 - y_full))
        - 12.0 * kappa * x_half * numpy.cos(kappa * (x_half**2 - y_full))
    )
    
    c_w = 2.0 * numpy.sin(kappa * y_full)
    c_s = kappa * numpy.cos(kappa * (x_full**2))
    c_e = 4.0 * kappa * numpy.cos(kappa * (y_full - 1.0)) - 2.0 * numpy.sin(kappa * (y_full - 1.0))
    c_n = -kappa * numpy.cos(kappa * (x_full**2 - 1.0))
    d_w = -numpy.sin(kappa * y_half)
    d_s = 2.0 * x_half * numpy.sin(kappa * x_half**2)
    d_e = -numpy.sin(kappa * (y_half - 1.0))
    d_n = 2.0 * x_half * numpy.sin(kappa * (x_half**2 - 1.0))
    
    return u_ana, v_ana, p_ana, (f_x, f_y), (c_w, c_s, c_e, c_n), (d_w, d_s, d_e, d_n)

    
def calc_rhs(size, fs, cs, ds):
    
    n, (f_x, f_y), (c_w, c_s, c_e, c_n), (d_w, d_s, d_e, d_n) = size, fs, cs, ds
    h = 1.0 / n
    
    c_x = f_x * h**2
    c_x[:, 0:1] += c_s * h
    c_x[:, -1:] += c_n * h
    c_x[0:1, :] += d_w
    c_x[-1:, :] += d_e
    
    c_y = f_y * h**2
    c_y[0:1, :] += c_w * h
    c_y[-1:, :] += c_e * h
    c_y[:, 0:1] += d_s
    c_y[:, -1:] += d_n
    
    c_i = numpy.zeros((n, n))
    c_i[0:1, :] -= d_w * h
    c_i[-1:, :] += d_e * h
    c_i[:, 0:1] -= d_s * h
    c_i[:, -1:] += d_n * h
    # Small modification to enforce consistency
    c_i -= c_i.mean()
    
    return c_x, c_y, c_i


def sum_res(size, u, v, p, c_x, c_y, c_i, num_threads=4, sup=False):
    n = size
    h = 1.0 / n
    res_x = op.wrapper_calc_res_x_norm(n, u, p, c_x, num_threads)
    if not sup:
        print("\tRes. of cons. momentum x: ", res_x)
    res_y = op.wrapper_calc_res_y_norm(n, v, p, c_y, num_threads)
    if not sup:
        print("\tRes. of cons. momentum y: ", res_y)
    res_i = op.wrapper_calc_res_i_norm(n, u, v, c_i, num_threads)
    if not sup:
        print("\tRes. of incomp.: ", res_i)
    return res_x, res_y, res_i


def sum_err(size, u_sol, v_sol, p_sol, u_ana, v_ana, p_ana, sup=False):
    n = size
    h = 1.0 / n
    err_u = numpy.linalg.norm((u_sol - u_ana).flat, 2.0) * h
    if not sup:
        print("\tError in u in L^2: {}".format(err_u))
    err_v = numpy.linalg.norm((v_sol - v_ana).flat, 2.0) * h
    if not sup:
        print("\tError in v in L^2: {}".format(err_v))
    err_p = numpy.linalg.norm(((p_sol - p_ana) - (p_sol.mean() - p_ana.mean())).flat, 2.0) * h
    if not sup:
        print("\tError in p in L^2: {}".format(err_p))
    return err_u, err_v, err_p
    