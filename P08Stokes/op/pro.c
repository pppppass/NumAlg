#include "op.h"

inline double calc_pro_close_p_step(int size, double* p_, int i, int j)
{
    int n = size;
    double (*p)[n/2] = (void*)p_;
    return p[i/2][j/2];
}

void add_pro_close_p(int size, double* p_, double* p_pro_, double m)
{
    int n = size;
    double (*p_pro)[n] = (void*)p_pro_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            p_pro[i][j] += m * calc_pro_close_p_step(n, p_, i, j);
    return ;
}

inline double calc_pro_bil_u_step(int size, double* u_, int i, int j)
{
    int n = size;
    double (*u)[n/2] = (void*)u_;
    int j_1 = 0, j_2 = 0;
    if (j > 0)
        j_1 = (j-1) / 2;
    else
        j_1 = 0;
    if (j < n-1)
        j_2 = (j+1) / 2;
    else
        j_2 = n/2-1;
    double u_1 = 0.0, u_2 = 0.0;
    if (i % 2 == 0)
    {
        if (i > 0)
            u_1 += u[i/2-1][j_1] / 2.0, u_2 += u[i/2-1][j_2] / 2.0;
        if (i < n-2)
            u_1 += u[i/2][j_1] / 2.0, u_2 += u[i/2][j_2] / 2.0;
    }
    else
        u_1 = u[i/2][j_1], u_2 = u[i/2][j_2];
    if (j % 2 == 0)
        return 1.0/4.0 * u_1 + 3.0/4.0 * u_2;
    else
        return 3.0/4.0 * u_1 + 1.0/4.0 * u_2;
}

void add_pro_bil_u(int size, double* u_, double* u_pro_, double m)
{
    int n = size;
    double (*u_pro)[n] = (void*)u_pro_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            u_pro[i][j] += m * calc_pro_bil_u_step(n, u_, i, j);
    return ;
}

inline double calc_pro_bil_v_step(int size, double* v_, int i, int j)
{
    int n = size;
    double (*v)[n/2-1] = (void*)v_;
    int i_1 = 0, i_2 = 0;
    if (i > 0)
        i_1 = (i-1) / 2;
    else
        i_1 = 0;
    if (i < n-1)
        i_2 = (i+1) / 2;
    else
        i_2 = n/2-1;
    double v_1 = 0.0, v_2 = 0.0;
    if (j % 2 == 0)
    {
        if (j > 0)
            v_1 += v[i_1][j/2-1] / 2.0, v_2 += v[i_2][j/2-1] / 2.0;
        if (j < n-2)
            v_1 += v[i_1][j/2] / 2.0, v_2 += v[i_2][j/2] / 2.0;
    }
    else
        v_1 = v[i_1][j/2], v_2 = v[i_2][j/2];
    if (i % 2 == 0)
        return 1.0/4.0 * v_1 + 3.0/4.0 * v_2;
    else
        return 3.0/4.0 * v_1 + 1.0/4.0 * v_2;
}

void add_pro_bil_v(int size, double* v_, double* v_pro_, double m)
{
    int n = size;
    double (*v_pro)[n-1] = (void*)v_pro_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            v_pro[i][j] += m * calc_pro_bil_v_step(n, v_, i, j);
    return ;
}
