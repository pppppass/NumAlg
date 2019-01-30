#include "op.h"

void sol_a_x_spec(int size, double* a_x_u_, double* u_, double* w_)
{
    int n = size;
    double (*a_x_u)[n] = (void*)a_x_u_, (*u)[n] = (void*)u_, (*w)[n] = (void*)w_;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            w[i][j] = a_x_u[i][j];
    
    int size_x[] = {n-1}, size_y[] = {n};
    fftw_r2r_kind kind_dst[] = {FFTW_RODFT00}, kind_dct_2[] = {FFTW_REDFT10}, kind_dct_3[] = {FFTW_REDFT01};

    fftw_plan
        plan_dst = fftw_plan_many_r2r(1, size_x, n, w_, NULL, n, 1, w_, NULL, n, 1, kind_dst, FFTW_ESTIMATE),
        plan_dct_2 = fftw_plan_many_r2r(1, size_y, n-1, w_, NULL, 1, n, w_, NULL, 1, n, kind_dct_2, FFTW_ESTIMATE),
        plan_dct_3 = fftw_plan_many_r2r(1, size_y, n-1, w_, NULL, 1, n, w_, NULL, 1, n, kind_dct_3, FFTW_ESTIMATE);

    fftw_execute(plan_dst);
    fftw_execute(plan_dct_2);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
        {
            double m_x = sin((1.0 + i) / n * M_PI / 2.0), m_y = sin((double)j / n * M_PI / 2.0);
            w[i][j] /= 16.0 * (m_x*m_x + m_y*m_y) * n*n;
        }

    fftw_execute(plan_dct_3);
    fftw_execute(plan_dst);

    fftw_destroy_plan(plan_dct_3);
    fftw_destroy_plan(plan_dct_2);
    fftw_destroy_plan(plan_dst);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            u[i][j] = w[i][j];

    return ;
}

void sol_a_y_spec(int size, double* a_y_v_, double* v_, double* w_)
{
    int n = size;
    double (*a_y_v)[n-1] = (void*)a_y_v_, (*v)[n-1] = (void*)v_, (*w)[n-1] = (void*)w_;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            w[i][j] = a_y_v[i][j];
    
    int size_x[] = {n}, size_y[] = {n-1};
    fftw_r2r_kind kind_dst[] = {FFTW_RODFT00}, kind_dct_2[] = {FFTW_REDFT10}, kind_dct_3[] = {FFTW_REDFT01};

    fftw_plan
        plan_dst = fftw_plan_many_r2r(1, size_y, n, w_, NULL, 1, n-1, w_, NULL, 1, n-1, kind_dst, FFTW_ESTIMATE),
        plan_dct_2 = fftw_plan_many_r2r(1, size_x, n-1, w_, NULL, n-1, 1, w_, NULL, n-1, 1, kind_dct_2, FFTW_ESTIMATE),
        plan_dct_3 = fftw_plan_many_r2r(1, size_x, n-1, w_, NULL, n-1, 1, w_, NULL, n-1, 1, kind_dct_3, FFTW_ESTIMATE);

    fftw_execute(plan_dst);
    fftw_execute(plan_dct_2);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
        {
            double m_x = sin((double)i / n * M_PI / 2.0), m_y = sin((1.0 + j) / n * M_PI / 2.0);
            w[i][j] /= 16.0 * (m_x*m_x + m_y*m_y) * n*n;
        }

    fftw_execute(plan_dct_3);
    fftw_execute(plan_dst);

    fftw_destroy_plan(plan_dct_3);
    fftw_destroy_plan(plan_dct_2);
    fftw_destroy_plan(plan_dst);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            v[i][j] = w[i][j];

    return ;
}
