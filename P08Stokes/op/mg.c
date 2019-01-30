#include "op.h"

void iter_sol_a_x_mg_gs(int size, int level, double* a_x_u_, double* u_, double* w_, int nu_down, int nu_up)
{
    int n = size, l = level;
    double* r_ = w_ + (n/2-1)*(n/2), * a_x_u_coar_ = w_, * u_coar_ = w_ + (n/2-1)*(n/2), * w_coar_ = w_ + 2*(n/2-1)*(n/2);
    double (*a_x_u)[n] = (void*)a_x_u_, (*r)[n] = (void*)r_, (*a_x_u_coar)[n/2] = (void*)a_x_u_coar_, (*u_coar)[n/2] = (void*)u_coar_;

    for (int i = 0; i < nu_down; i++)
        iter_sol_a_x_gs(n, a_x_u_, u_);
    
    if (l > 0)
    {
        // cblas_dcopy((n-1)*n, a_x_u_, 1, r_, 1);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n-1; i++)
            for (int j = 0; j < n; j++)
                r[i][j] = a_x_u[i][j];
        add_a_x(n, u_, r_, -1.0);
        
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n/2-1; i++)
            for (int j = 0; j < n/2; j++)
                a_x_u_coar[i][j] = 0.0;
        add_res_close_u(n, r_, a_x_u_coar_, 4.0);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < n/2-1; i++)
            for (int j = 0; j < n/2; j++)
                 u_coar[i][j] = 0.0;
        iter_sol_a_x_mg_gs(n/2, l-1, a_x_u_coar_, u_coar_, w_coar_, nu_down, nu_up);

        add_pro_bil_u(n, u_coar_, u_, 1.0);
    }

    for (int i = 0; i < nu_up; i++)
        iter_sol_a_x_gs(n, a_x_u_, u_);
    
    return ;
}

void iter_sol_a_y_mg_gs(int size, int level, double* a_y_v_, double* v_, double* w_, int nu_down, int nu_up)
{
    int n = size, l = level;
    double* r_ = w_ + (n/2)*(n/2-1), * a_y_v_coar_ = w_, * v_coar_ = w_ + (n/2)*(n/2-1), * w_coar_ = w_ + 2*(n/2)*(n/2-1);
    double (*a_y_v)[n-1] = (void*)a_y_v_, (*r)[n-1] = (void*)r_, (*a_y_v_coar)[n/2-1] = (void*)a_y_v_coar_, (*v_coar)[n/2-1] = (void*)v_coar_;

    for (int i = 0; i < nu_down; i++)
        iter_sol_a_y_gs(n, a_y_v_, v_);
    
    if (l > 0)
    {
        // cblas_dcopy(n*(n-1), a_y_v_, 1, r_, 1);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            r[i][j] = a_y_v[i][j];
        add_a_y(n, v_, r_, -1.0);
        
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n/2; i++)
            for (int j = 0; j < n/2-1; j++)
                a_y_v_coar[i][j] = 0.0;
        add_res_close_v(n, r_, a_y_v_coar_, 4.0);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < n/2; i++)
            for (int j = 0; j < n/2-1; j++)
                 v_coar[i][j] = 0.0;
        iter_sol_a_y_mg_gs(n/2, l-1, a_y_v_coar_, v_coar_, w_coar_, nu_down, nu_up);

        add_pro_bil_v(n, v_coar_, v_, 1.0);
    }

    for (int i = 0; i < nu_up; i++)
        iter_sol_a_y_gs(n, a_y_v_, v_);
    
    return ;
}
