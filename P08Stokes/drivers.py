import time
import numpy
import scipy.sparse.linalg
import models
import gs, dgs, res, pro, mat, spec
import op


def iter_dgs_mg(size, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, sol_func, num_threads):
        
    n, l = size, level
    
    for _ in range(nu_down):
        op.wrapper_iter_gs_u(n, u, p, c_x, num_threads)
        op.wrapper_iter_gs_v(n, v, p, c_y, num_threads)
        op.wrapper_iter_dgs_incomp(n, u, v, p, c_i, num_threads)
    
    if level > 0:

        r_x = op.wrapper_add_b_x(n, p, op.wrapper_add_a_x(n, u, c_x.copy(), -1.0, num_threads), -1.0, num_threads)
        r_y = op.wrapper_add_b_y(n, p, op.wrapper_add_a_y(n, v, c_y.copy(), -1.0, num_threads), -1.0, num_threads)
        r_i = op.wrapper_add_b_y_t(n, v, op.wrapper_add_b_x_t(n, u, c_i.copy(), -1.0, num_threads), -1.0, num_threads)
        
        # Multiply by 4 here because we balance the matrix with *= h^2
        r_x_coar = op.wrapper_add_res_close_u(n, r_x, numpy.zeros((n//2-1, n//2)), 4.0, num_threads)
        r_y_coar = op.wrapper_add_res_close_v(n, r_y, numpy.zeros((n//2, n//2-1)), 4.0, num_threads)
        r_i_coar = op.wrapper_add_res_close_p(n, r_i, numpy.zeros((n//2, n//2)), 4.0, num_threads)
        
        del r_x, r_y, r_i
        
        u_coar = numpy.zeros((n//2-1, n//2))
        v_coar = numpy.zeros((n//2, n//2-1))
        p_coar = numpy.zeros((n//2, n//2))
        
        iter_dgs_mg(n//2, l-1, nu_down, nu_up, u_coar, v_coar, p_coar, r_x_coar, r_y_coar, r_i_coar, sol_func, num_threads)
        
        del r_x_coar, r_y_coar, r_i_coar
        
        op.wrapper_add_pro_bil_u(n, u_coar, u, 1.0, num_threads)
        op.wrapper_add_pro_bil_v(n, v_coar, v, 1.0, num_threads)
        op.wrapper_add_pro_close_p(n, p_coar, p, 1.0, num_threads)
        
        del u_coar, v_coar, p_coar
    
    elif sol_func is not None:
        u[:, :], v[:, :], p[:, :] = sol_func(c_x, c_y, c_i)
    
    for _ in range(nu_up):
        op.wrapper_iter_gs_u(n, u, p, c_x, num_threads)
        op.wrapper_iter_gs_v(n, v, p, c_y, num_threads)
        op.wrapper_iter_dgs_incomp(n, u, v, p, c_i, num_threads)
    
    return u, v, p


def driver_dgs_mg(size, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps, iter_max, spsolve, num_threads, sup=False, sup_sub=True):
    start = time.time()
    n, l = size, level
    if spsolve:
        n_fin = n // (2**l)
        # Convert to CSC because SuperLU needs that
        a = mat.get_mat(n_fin).tocsc()
        a_lu = scipy.sparse.linalg.splu(a)
        sol_func = lambda r_x, r_y, r_i: mat.sol_dir(n_fin, lambda b: a_lu.solve(b), r_x, r_y, r_i)
    else:
        sol_func = None
    for ctr in range(iter_max):
        iter_dgs_mg(size, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, sol_func, num_threads)
        if ctr % 1 == 0:
            if not sup_sub:
                print("Step {}:".format(ctr))
            res_x, res_y, res_i = models.sum_res(n, u, v, p, c_x, c_y, c_i, sup=sup_sub)
            if res_x < eps and res_y < eps and res_i < eps:
                break
    end = time.time()
    ctr += 1
    elapsed = end - start
    if not sup:
        print("Summary:")
        print("\tIterations: {}".format(ctr))
        print("\tElapsed: {}".format(elapsed))
    return u, v, p, ctr, elapsed


def iter_uzawa_mg(size, alpha, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, num_threads, fftw=True):
        
    n, l = size, level
    
    for _ in range(nu_down):
        if fftw:
            op.wrapper_sol_a_x_spec(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads), u, num_threads)
            op.wrapper_sol_a_y_spec(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads), v, num_threads)
        else:
            u[:, :] = spec.sol_a_x_spec(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads))
            v[:, :] = spec.sol_a_y_spec(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads))
        op.wrapper_add_b_x_t(n, u, p, alpha * n**2, num_threads)
        op.wrapper_add_b_y_t(n, v, p, alpha * n**2, num_threads)
        # Update when c_i is not zero
        p -= alpha * n**2 * c_i
    
    if level > 0:

        r_x = op.wrapper_add_b_x(n, p, op.wrapper_add_a_x(n, u, c_x.copy(), -1.0, num_threads), -1.0, num_threads)
        r_y = op.wrapper_add_b_y(n, p, op.wrapper_add_a_y(n, v, c_y.copy(), -1.0, num_threads), -1.0, num_threads)
        r_i = op.wrapper_add_b_y_t(n, v, op.wrapper_add_b_x_t(n, u, c_i.copy(), -1.0, num_threads), -1.0, num_threads)
        
        # Multiply by 4 here because we balance the matrix with *= h^2
        r_x_coar = op.wrapper_add_res_close_u(n, r_x, numpy.zeros((n//2-1, n//2)), 4.0, num_threads)
        r_y_coar = op.wrapper_add_res_close_v(n, r_y, numpy.zeros((n//2, n//2-1)), 4.0, num_threads)
        r_i_coar = op.wrapper_add_res_close_p(n, r_i, numpy.zeros((n//2, n//2)), 4.0, num_threads)
        
        del r_x, r_y, r_i
        
        u_coar = numpy.zeros((n//2-1, n//2))
        v_coar = numpy.zeros((n//2, n//2-1))
        p_coar = numpy.zeros((n//2, n//2))
        
        iter_uzawa_mg(n//2, alpha, l-1, nu_down, nu_up, u_coar, v_coar, p_coar, r_x_coar, r_y_coar, r_i_coar, num_threads, fftw=fftw)
        
        del r_x_coar, r_y_coar, r_i_coar
        
        op.wrapper_add_pro_bil_u(n, u_coar, u, 1.0, num_threads)
        op.wrapper_add_pro_bil_v(n, v_coar, v, 1.0, num_threads)
        op.wrapper_add_pro_close_p(n, p_coar, p, 1.0, num_threads)
        
        del u_coar, v_coar, p_coar
    
    for _ in range(nu_up):
        if fftw:
            op.wrapper_sol_a_x_spec(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads), u, num_threads)
            op.wrapper_sol_a_y_spec(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads), v, num_threads)
        else:
            u[:, :] = spec.sol_a_x_spec(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads))
            v[:, :] = spec.sol_a_y_spec(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads))
        op.wrapper_add_b_x_t(n, u, p, alpha * n**2, num_threads)
        op.wrapper_add_b_y_t(n, v, p, alpha * n**2, num_threads)
        p -= alpha * n**2 * c_i
    
    return u, v, p


def driver_uzawa_mg(size, alpha, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps, iter_max, num_threads, sup=False, sup_sub=True, fftw=True):
    start = time.time()
    n, l = size, level
    for ctr in range(iter_max):
        iter_uzawa_mg(size, alpha, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, num_threads, fftw=fftw)
        if ctr % 1 == 0:
            if not sup_sub:
                print("Step {}:".format(ctr))
            res_x, res_y, res_i = models.sum_res(n, u, v, p, c_x, c_y, c_i, sup=sup_sub)
            if res_x < eps and res_y < eps and res_i < eps:
                break
    end = time.time()
    ctr += 1
    elapsed = end - start
    if not sup:
        print("Summary:")
        print("\tIterations: {}".format(ctr))
        print("\tElapsed: {}".format(elapsed))
    return u, v, p, ctr, elapsed


def iter_uzawa_inexact_cg_mg(size, alpha, eps_cg, iter_cg, tau, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, num_threads):
        
    n, l = size, level
    
    for _ in range(nu_down):
        eps_norm = op.wrapper_calc_res_i_norm(n, u, v, c_i, num_threads)
        eps_cg_real = max(tau * eps_norm, eps_cg)
        op.wrapper_sol_a_x_cg(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads), u, eps_cg_real, iter_cg, num_threads)
        op.wrapper_sol_a_y_cg(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads), v, eps_cg_real, iter_cg, num_threads)
        op.wrapper_add_b_x_t(n, u, p, alpha * n**2, num_threads)
        op.wrapper_add_b_y_t(n, v, p, alpha * n**2, num_threads)
        # Update when c_i is not zero
        p -= alpha * n**2 * c_i
    
    if level > 0:

        r_x = op.wrapper_add_b_x(n, p, op.wrapper_add_a_x(n, u, c_x.copy(), -1.0, num_threads), -1.0, num_threads)
        r_y = op.wrapper_add_b_y(n, p, op.wrapper_add_a_y(n, v, c_y.copy(), -1.0, num_threads), -1.0, num_threads)
        r_i = op.wrapper_add_b_y_t(n, v, op.wrapper_add_b_x_t(n, u, c_i.copy(), -1.0, num_threads), -1.0, num_threads)
        
        # Multiply by 4 here because we balance the matrix with *= h^2
        r_x_coar = op.wrapper_add_res_close_u(n, r_x, numpy.zeros((n//2-1, n//2)), 4.0, num_threads)
        r_y_coar = op.wrapper_add_res_close_v(n, r_y, numpy.zeros((n//2, n//2-1)), 4.0, num_threads)
        r_i_coar = op.wrapper_add_res_close_p(n, r_i, numpy.zeros((n//2, n//2)), 4.0, num_threads)
        
        del r_x, r_y, r_i
        
        u_coar = numpy.zeros((n//2-1, n//2))
        v_coar = numpy.zeros((n//2, n//2-1))
        p_coar = numpy.zeros((n//2, n//2))
        
        iter_uzawa_inexact_cg_mg(n//2, alpha, eps_cg, iter_cg, tau, l-1, nu_down, nu_up, u_coar, v_coar, p_coar, r_x_coar, r_y_coar, r_i_coar, num_threads)
        
        del r_x_coar, r_y_coar, r_i_coar
        
        op.wrapper_add_pro_bil_u(n, u_coar, u, 1.0, num_threads)
        op.wrapper_add_pro_bil_v(n, v_coar, v, 1.0, num_threads)
        op.wrapper_add_pro_close_p(n, p_coar, p, 1.0, num_threads)
        
        del u_coar, v_coar, p_coar
    
    for _ in range(nu_up):
        eps_norm = op.wrapper_calc_res_i_norm(n, u, v, c_i, num_threads)
        eps_cg_real = max(tau * eps_norm, eps_cg)
        op.wrapper_sol_a_x_cg(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads), u, eps_cg_real, iter_cg, num_threads)
        op.wrapper_sol_a_y_cg(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads), v, eps_cg_real, iter_cg, num_threads)
        op.wrapper_add_b_x_t(n, u, p, alpha * n**2, num_threads)
        op.wrapper_add_b_y_t(n, v, p, alpha * n**2, num_threads)
        p -= alpha * n**2 * c_i
    
    return u, v, p


def driver_uzawa_inexact_cg_mg(size, alpha, eps_cg, iter_cg, tau, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps, iter_max, num_threads, sup=False, sup_sub=True):
    start = time.time()
    n, l = size, level
    for ctr in range(iter_max):
        iter_uzawa_inexact_cg_mg(size, alpha, eps_cg, iter_cg, tau, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, num_threads)
        if ctr % 1 == 0:
            if not sup_sub:
                print("Step {}:".format(ctr))
            res_x, res_y, res_i = models.sum_res(n, u, v, p, c_x, c_y, c_i, sup=sup_sub)
            if res_x < eps and res_y < eps and res_i < eps:
                break
    end = time.time()
    ctr += 1
    elapsed = end - start
    if not sup:
        print("Summary:")
        print("\tIterations: {}".format(ctr))
        print("\tElapsed: {}".format(elapsed))
    return u, v, p, ctr, elapsed


def iter_uzawa_inexact_pcg_mg_gs_mg(size, alpha, eps_cg, iter_cg, tau, level_pcg, nu_down_pcg, nu_up_pcg, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, num_threads):
        
    n, l = size, level
    
    for _ in range(nu_down):
        eps_norm = op.wrapper_calc_res_i_norm(n, u, v, c_i, num_threads)
        eps_cg_real = max(tau * eps_norm, eps_cg)
        op.wrapper_sol_a_x_pcg_mg_gs(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads), u, eps_cg_real, iter_cg, level_pcg, nu_down_pcg, nu_up_pcg, num_threads)
        op.wrapper_sol_a_y_pcg_mg_gs(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads), v, eps_cg_real, iter_cg, level_pcg, nu_down_pcg, nu_up_pcg, num_threads)
        op.wrapper_add_b_x_t(n, u, p, alpha * n**2, num_threads)
        op.wrapper_add_b_y_t(n, v, p, alpha * n**2, num_threads)
        # Update when c_i is not zero
        p -= alpha * n**2 * c_i
    
    if level > 0:

        r_x = op.wrapper_add_b_x(n, p, op.wrapper_add_a_x(n, u, c_x.copy(), -1.0, num_threads), -1.0, num_threads)
        r_y = op.wrapper_add_b_y(n, p, op.wrapper_add_a_y(n, v, c_y.copy(), -1.0, num_threads), -1.0, num_threads)
        r_i = op.wrapper_add_b_y_t(n, v, op.wrapper_add_b_x_t(n, u, c_i.copy(), -1.0, num_threads), -1.0, num_threads)
        
        # Multiply by 4 here because we balance the matrix with *= h^2
        r_x_coar = op.wrapper_add_res_close_u(n, r_x, numpy.zeros((n//2-1, n//2)), 4.0, num_threads)
        r_y_coar = op.wrapper_add_res_close_v(n, r_y, numpy.zeros((n//2, n//2-1)), 4.0, num_threads)
        r_i_coar = op.wrapper_add_res_close_p(n, r_i, numpy.zeros((n//2, n//2)), 4.0, num_threads)
        
        del r_x, r_y, r_i
        
        u_coar = numpy.zeros((n//2-1, n//2))
        v_coar = numpy.zeros((n//2, n//2-1))
        p_coar = numpy.zeros((n//2, n//2))
        
        iter_uzawa_inexact_pcg_mg_gs_mg(n//2, alpha, eps_cg, iter_cg, tau, level_pcg-1, nu_down_pcg, nu_up_pcg, l-1, nu_down, nu_up, u_coar, v_coar, p_coar, r_x_coar, r_y_coar, r_i_coar, num_threads)
        
        del r_x_coar, r_y_coar, r_i_coar
        
        op.wrapper_add_pro_bil_u(n, u_coar, u, 1.0, num_threads)
        op.wrapper_add_pro_bil_v(n, v_coar, v, 1.0, num_threads)
        op.wrapper_add_pro_close_p(n, p_coar, p, 1.0, num_threads)
        
        del u_coar, v_coar, p_coar
    
    for _ in range(nu_up):
        eps_norm = op.wrapper_calc_res_i_norm(n, u, v, c_i, num_threads)
        eps_cg_real = max(tau * eps_norm, eps_cg)
        op.wrapper_sol_a_x_pcg_mg_gs(n, op.wrapper_add_b_x(n, p, c_x.copy(), -1.0, num_threads), u, eps_cg_real, iter_cg, level_pcg, nu_down_pcg, nu_up_pcg, num_threads)
        op.wrapper_sol_a_y_pcg_mg_gs(n, op.wrapper_add_b_y(n, p, c_y.copy(), -1.0, num_threads), v, eps_cg_real, iter_cg, level_pcg, nu_down_pcg, nu_up_pcg, num_threads)
        op.wrapper_add_b_x_t(n, u, p, alpha * n**2, num_threads)
        op.wrapper_add_b_y_t(n, v, p, alpha * n**2, num_threads)
        p -= alpha * n**2 * c_i
    
    return u, v, p


def driver_uzawa_inexact_pcg_mg_gs_mg(size, alpha, eps_cg, iter_cg, tau, level_pcg, nu_down_pcg, nu_up_pcg, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps, iter_max, num_threads, sup=False, sup_sub=True):
    start = time.time()
    n, l = size, level
    for ctr in range(iter_max):
        iter_uzawa_inexact_pcg_mg_gs_mg(size, alpha, eps_cg, iter_cg, tau, level_pcg, nu_down_pcg, nu_up_pcg, level, nu_down, nu_up, u, v, p, c_x, c_y, c_i, num_threads)
        if ctr % 1 == 0:
            if not sup_sub:
                print("Step {}:".format(ctr))
            res_x, res_y, res_i = models.sum_res(n, u, v, p, c_x, c_y, c_i, sup=sup_sub)
            if res_x < eps and res_y < eps and res_i < eps:
                break
    end = time.time()
    ctr += 1
    elapsed = end - start
    if not sup:
        print("Summary:")
        print("\tIterations: {}".format(ctr))
        print("\tElapsed: {}".format(elapsed))
    return u, v, p, ctr, elapsed
