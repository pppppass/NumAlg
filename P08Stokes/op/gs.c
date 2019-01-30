#include "op.h"

inline void iter_gs_u_step(int size, double* u_, double* p_, double* c_x_, int i, int j)
{
    int n = size;
    double h = 1.0 / n;
    double (*u)[n] = (void*)u_, (*p)[n] = (void*)p_, (*c_x)[n] = (void*)c_x_;
    if (i == 0)
    {
        if (j == 0)
            u[0][0] = (c_x[0][0] - (p[1][0] - p[0][0]) * h + u[1][0] + u[0][1]) / 3.0;
        else if (j == n-1)
            u[0][n-1] = (c_x[0][n-1] - (p[1][n-1] - p[0][n-1]) * h + u[1][n-1] + u[0][n-2]) / 3.0;
        else
            u[0][j] = (c_x[0][j] - (p[1][j] - p[0][j]) * h + u[1][j] + u[0][j-1] + u[0][j+1]) / 4.0;
    }
    else if (i == n-2)
    {
        if (j == 0)
            u[n-2][0] = (c_x[n-2][0] - (p[n-1][0] - p[n-2][0]) * h + u[n-3][0] + u[n-2][1]) / 3.0;
        else if (j == n-1)
            u[n-2][n-1] = (c_x[n-2][n-1] - (p[n-1][n-1] - p[n-2][n-1]) * h + u[n-3][n-1] + u[n-2][n-2]) / 3.0;
        else
            u[n-2][j] = (c_x[n-2][j] - (p[n-1][j] - p[n-2][j]) * h + u[n-3][j] + u[n-2][j-1] + u[n-2][j+1]) / 4.0;
    }
    else
    {
        if (j == 0)
            u[i][0] = (c_x[i][0] - (p[i+1][0] - p[i][0]) * h + u[i-1][0] + u[i+1][0] + u[i][1]) / 3.0;
        else if (j == n-1)
            u[i][n-1] = (c_x[i][n-1] - (p[i+1][n-1] - p[i][n-1]) * h + u[i-1][n-1] + u[i+1][n-1] + u[i][n-2]) / 3.0;
        else
            u[i][j] = (c_x[i][j] - (p[i+1][j] - p[i][j]) * h + u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / 4.0;
    }
    return ;
}

void iter_gs_u(int size, double* u_, double* p_, double* c_x_)
{
    int n = size;
#pragma omp parallel for collapse(2) if(n >= SIZE_MIN_PARALLEL)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            iter_gs_u_step(n, u_, p_, c_x_, i, j);
    return ;
}

inline void iter_gs_v_step(int size, double* v_, double* p_, double* c_y_, int i, int j)
{
    int n = size;
    double h = 1.0 / n;
    double (*v)[n-1] = (void*)v_, (*p)[n] = (void*)p_, (*c_y)[n-1] = (void*)c_y_;
    if (i == 0)
    {
        if (j == 0)
            v[0][0] = (c_y[0][0] - (p[0][1] - p[0][0]) * h + v[1][0] + v[0][1]) / 3.0;
        else if (j == n-2)
            v[0][n-2] = (c_y[0][n-2] - (p[0][n-1] - p[0][n-2]) * h + v[1][n-2] + v[0][n-3]) / 3.0;
        else
            v[0][j] = (c_y[0][j] - (p[0][j+1] - p[0][j]) * h + v[1][j] + v[0][j-1] + v[0][j+1]) / 3.0;
    }
    else if (i == n-1)
    {
        if (j == 0)
            v[n-1][0] = (c_y[n-1][0] - (p[n-1][1] - p[n-1][0]) * h + v[n-2][0] + v[n-1][1]) / 3.0;
        else if (j == n-2)
            v[n-1][n-2] = (c_y[n-1][n-2] - (p[n-1][n-1] - p[n-1][n-2]) * h + v[n-2][n-2] + v[n-1][n-3]) / 3.0;
        else
            v[n-1][j] = (c_y[n-1][j] - (p[n-1][j+1] - p[n-1][j]) * h + v[n-2][j] + v[n-1][j-1] + v[n-1][j+1]) / 3.0;
    }
    else
    {
        if (j == 0)
            v[i][0] = (c_y[i][0] - (p[i][1] - p[i][0]) * h + v[i-1][0] + v[i+1][0] + v[i][1]) / 4.0;
        else if (j == n-2)
            v[i][n-2] = (c_y[i][n-2] - (p[i][n-1] - p[i][n-2]) * h + v[i-1][n-2] + v[i+1][n-2] + v[i][n-3]) / 4.0;
        else
            v[i][j] = (c_y[i][j] - (p[i][j+1] - p[i][j]) * h + v[i-1][j] + v[i+1][j] + v[i][j-1] + v[i][j+1]) / 4.0;
    }
}

void iter_gs_v(int size, double* v_, double* p_, double* c_y_)
{
    int n = size;
#pragma omp parallel for collapse(2) if(n >= SIZE_MIN_PARALLEL)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            iter_gs_v_step(n, v_, p_, c_y_, i, j);
    return ;
}

inline void iter_sol_a_x_gs_step(int size, double* a_x_u_, double* u_, int i, int j)
{
    int n = size;
    // double h = 1.0 / n;
    double (*a_x_u)[n] = (void*)a_x_u_, (*u)[n] = (void*)u_;
    if (i == 0)
    {
        if (j == 0)
            u[0][0] = (a_x_u[0][0] + u[1][0] + u[0][1]) / 3.0;
        else if (j == n-1)
            u[0][n-1] = (a_x_u[0][n-1] + u[1][n-1] + u[0][n-2]) / 3.0;
        else
            u[0][j] = (a_x_u[0][j] + u[1][j] + u[0][j-1] + u[0][j+1]) / 4.0;
    }
    else if (i == n-2)
    {
        if (j == 0)
            u[n-2][0] = (a_x_u[n-2][0] + u[n-3][0] + u[n-2][1]) / 3.0;
        else if (j == n-1)
            u[n-2][n-1] = (a_x_u[n-2][n-1] + u[n-3][n-1] + u[n-2][n-2]) / 3.0;
        else
            u[n-2][j] = (a_x_u[n-2][j] + u[n-3][j] + u[n-2][j-1] + u[n-2][j+1]) / 4.0;
    }
    else
    {
        if (j == 0)
            u[i][0] = (a_x_u[i][0] + u[i-1][0] + u[i+1][0] + u[i][1]) / 3.0;
        else if (j == n-1)
            u[i][n-1] = (a_x_u[i][n-1] + u[i-1][n-1] + u[i+1][n-1] + u[i][n-2]) / 3.0;
        else
            u[i][j] = (a_x_u[i][j] + u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / 4.0;
    }
    return ;
}

void iter_sol_a_x_gs(int size, double* a_x_u_, double* u_)
{
    int n = size;
#pragma omp parallel for collapse(2) if(n >= SIZE_MIN_PARALLEL)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            iter_sol_a_x_gs_step(n, a_x_u_, u_, i, j);
    return ;
}

inline void iter_sol_a_y_gs_step(int size, double* a_y_v_, double* v_, int i, int j)
{
    int n = size;
    // double h = 1.0 / n;
    double (*a_y_v)[n-1] = (void*)a_y_v_, (*v)[n-1] = (void*)v_;
    if (i == 0)
    {
        if (j == 0)
            v[0][0] = (a_y_v[0][0] + v[1][0] + v[0][1]) / 3.0;
        else if (j == n-2)
            v[0][n-2] = (a_y_v[0][n-2] + v[1][n-2] + v[0][n-3]) / 3.0;
        else
            v[0][j] = (a_y_v[0][j] + v[1][j] + v[0][j-1] + v[0][j+1]) / 3.0;
    }
    else if (i == n-1)
    {
        if (j == 0)
            v[n-1][0] = (a_y_v[n-1][0] + v[n-2][0] + v[n-1][1]) / 3.0;
        else if (j == n-2)
            v[n-1][n-2] = (a_y_v[n-1][n-2] + v[n-2][n-2] + v[n-1][n-3]) / 3.0;
        else
            v[n-1][j] = (a_y_v[n-1][j] + v[n-2][j] + v[n-1][j-1] + v[n-1][j+1]) / 3.0;
    }
    else
    {
        if (j == 0)
            v[i][0] = (a_y_v[i][0] + v[i-1][0] + v[i+1][0] + v[i][1]) / 4.0;
        else if (j == n-2)
            v[i][n-2] = (a_y_v[i][n-2] + v[i-1][n-2] + v[i+1][n-2] + v[i][n-3]) / 4.0;
        else
            v[i][j] = (a_y_v[i][j] + v[i-1][j] + v[i+1][j] + v[i][j-1] + v[i][j+1]) / 4.0;
    }
}

void iter_sol_a_y_gs(int size, double* a_y_v_, double* v_)
{
    int n = size;
#pragma omp parallel for collapse(2) if(n >= SIZE_MIN_PARALLEL)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            iter_sol_a_y_gs_step(n, a_y_v_, v_, i, j);
    return ;
}
