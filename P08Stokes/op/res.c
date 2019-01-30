#include "op.h"

inline double calc_res_close_p_step(int size, double* p_, int i, int j)
{
    int n = size;
    double (*p)[n] = (void*)p_;
    return (p[i*2][j*2] + p[i*2][j*2+1] + p[i*2+1][j*2] + p[i*2+1][j*2+1]) / 4.0;
}

void add_res_close_p(int size, double* p_, double* p_res_, double m)
{
    int n = size;
    double (*p_res)[n/2] = (void*)p_res_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n/2; i++)
        for (int j = 0; j < n/2; j++)
            p_res[i][j] += m * calc_res_close_p_step(n, p_, i, j);
    return ;
}

inline double calc_res_close_u_step(int size, double* u_, int i, int j)
{
    int n = size;
    double (*u)[n] = (void*)u_;
    return (u[i*2][j*2] + 2.0 * u[i*2+1][j*2] + u[i*2+2][j*2] + u[i*2][j*2+1] + 2.0 * u[i*2+1][j*2+1] + u[i*2+2][j*2+1]) / 8.0;
}

void add_res_close_u(int size, double* u_, double* u_res_, double m)
{
    int n = size;
    double (*u_res)[n/2] = (void*)u_res_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n/2-1; i++)
        for (int j = 0; j < n/2; j++)
            u_res[i][j] += m * calc_res_close_u_step(n, u_, i, j);
    return ;
}

inline double calc_res_close_v_step(int size, double* v_, int i, int j)
{
    int n = size;
    double (*v)[n-1] = (void*)v_;
    return (v[i*2][j*2] + 2.0 * v[i*2][j*2+1] + v[i*2][j*2+2] + v[i*2+1][j*2] + 2.0 * v[i*2+1][j*2+1] + v[i*2+1][j*2+2]) / 8.0;
}

void add_res_close_v(int size, double* v_, double* v_res_, double m)
{
    int n = size;
    double (*v_res)[n/2-1] = (void*)v_res_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n/2; i++)
        for (int j = 0; j < n/2-1; j++)
            v_res[i][j] += m * calc_res_close_v_step(n, v_, i, j);
    return ;
}
