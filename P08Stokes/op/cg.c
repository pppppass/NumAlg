#include "op.h"

int sol_a_x_cg(int size, double* a_x_u_, double* u_, double* w_, double eps, int iter_max)
{
    int n = size;
    double (*b)[n] = (void*)a_x_u_, (*x)[n] = (void*)u_, (*r)[n] = (void*)w_, (*p)[n] = (void*)(w_+(n-1)*n), (*w)[n] = (void*)(w_+2*(n-1)*n);
    double rho = 0.0, rho_old = 0.0;
    
    // cblas_dcopy((n-1)*n, (void*)b, 1, (void*)r, 1);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            r[i][j] = b[i][j];
    add_a_x(n, (void*)x, (void*)r, -1.0);
    
    // rho = cblas_ddot((n-1)*n, (void*)r, 1, (void*)r, 1);
    rho = 0.0;
#pragma omp parallel for collapse(2) reduction(+: rho)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            rho += r[i][j] * r[i][j];

    int ctr = 0;
    while (ctr < iter_max && rho >= eps*eps)
    {

        ctr++;
        if (ctr == 1)
            // cblas_dcopy((n-1)*n, (void*)r, 1, (void*)p, 1);
        {
#pragma omp parallel for collapse(2)
            for (int i = 0; i < n-1; i++)
                for (int j = 0; j < n; j++)
                    p[i][j] = r[i][j];
        }
        else
        {
            double beta = rho / rho_old;
            // cblas_dscal((n-1)*n, beta, (void*)p, 1);
            // cblas_daxpy((n-1)*n, 1.0, (void*)r, 1, (void*)p, 1);
#pragma omp parallel for collapse(2)
            for (int i = 0; i < n-1; i++)
                for (int j = 0; j < n; j++)
                    p[i][j] = r[i][j] + beta * p[i][j];
        }

        // No function to set zero
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n-1; i++)
            for (int j = 0; j < n; j++)
                w[i][j] = 0.0;
        add_a_x(n, (void*)p, (void*)w, 1.0);

        // double alpha = rho / cblas_ddot((n-1)*n, (void*)p, 1, (void*)w, 1);
        double alpha = 0.0;
#pragma omp parallel for collapse(2) reduction(+: alpha)
        for (int i = 0; i < n-1; i++)
            for (int j = 0; j < n; j++)
                alpha += p[i][j] * w[i][j];
        alpha = rho / alpha;
        
        // cblas_daxpy((n-1)*n, alpha, (void*)p, 1, (void*)x, 1);
        // cblas_daxpy((n-1)*n, -alpha, (void*)w, 1, (void*)r, 1);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < n-1; i++)
            for (int j = 0; j < n; j++)
            {
                x[i][j] += alpha * p[i][j];
                r[i][j] -= alpha * w[i][j];
            }
        
        rho_old = rho;
        // rho = cblas_ddot((n-1)*n, (void*)r, 1, (void*)r, 1);
        rho = 0.0;
#pragma omp parallel for collapse(2) reduction(+: rho)
        for (int i = 0; i < n-1; i++)
            for (int j = 0; j < n; j++)
                rho += r[i][j]*r[i][j];
    }

    return ctr;
}

int sol_a_y_cg(int size, double* a_y_v_, double* v_, double* w_, double eps, int iter_max)
{
    int n = size;
    double (*b)[n-1] = (void*)a_y_v_, (*x)[n-1] = (void*)v_, (*r)[n-1] = (void*)w_, (*p)[n-1] = (void*)(w_+n*(n-1)), (*w)[n-1] = (void*)(w_+2*n*(n-1));
    double rho = 0.0, rho_old = 0.0;
    
    // cblas_dcopy(n*(n-1), (void*)b, 1, (void*)r, 1);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            r[i][j] = b[i][j];
    add_a_y(n, (void*)x, (void*)r, -1.0);
    
    // rho = cblas_ddot(n*(n-1), (void*)r, 1, (void*)r, 1);
    rho = 0.0;
#pragma omp parallel for collapse(2) reduction(+: rho)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            rho += r[i][j]*r[i][j];

    int ctr = 0;
    while (ctr < iter_max && rho >= eps*eps)
    {

        ctr++;
        if (ctr == 1)
            // cblas_dcopy(n*(n-1), (void*)r, 1, (void*)p, 1);
        {
#pragma omp parallel for collapse(2)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n-1; j++)
                    p[i][j] = r[i][j];
        }
        else
        {
            double beta = rho / rho_old;
            // cblas_dscal(n*(n-1), beta, (void*)p, 1);
            // cblas_daxpy(n*(n-1), 1.0, (void*)r, 1, (void*)p, 1);
#pragma omp parallel for collapse(2)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n-1; j++)
                    p[i][j] = r[i][j] + beta * p[i][j];
        }

        // No function to set zero
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n-1; j++)
                w[i][j] = 0.0;
        add_a_y(n, (void*)p, (void*)w, 1.0);

        // double alpha = rho / cblas_ddot(n*(n-1), (void*)p, 1, (void*)w, 1);
        double alpha = 0.0;
#pragma omp parallel for collapse(2) reduction(+: alpha)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n-1; j++)
                alpha += p[i][j] * w[i][j];
        alpha = rho / alpha;
        
        // cblas_daxpy(n*(n-1), alpha, (void*)p, 1, (void*)x, 1);
        // cblas_daxpy(n*(n-1), -alpha, (void*)w, 1, (void*)r, 1);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n-1; j++)
            {
                x[i][j] += alpha * p[i][j];
                r[i][j] -= alpha * w[i][j];
            }
        
        rho_old = rho;
        // rho = cblas_ddot(n*(n-1), (void*)r, 1, (void*)r, 1);
        rho = 0.0;
#pragma omp parallel for collapse(2) reduction(+: rho)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n-1; j++)
                rho += r[i][j]*r[i][j];
    }

    return ctr;
}
