#include "op.h"

inline double calc_b_x_t_step(int size, double* u_, int i, int j)
{
    int n = size;
    double h = 1.0 / n;
    double (*u)[n] = (void*)u_;
    double r = 0.0;
    if (i > 0)
        r += u[i-1][j] * h;
    if (i < n-1)
        r -= u[i][j] * h;
    return r;
}

void add_b_x_t(int size, double* u_, double* r_i_, double m)
{
    int n = size;
    double (*r_i)[n] = (void*)r_i_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            r_i[i][j] += m * calc_b_x_t_step(n, u_, i, j);
    return ;
}

inline double calc_b_y_t_step(int size, double* v_, int i, int j)
{
    int n = size;
    double h = 1.0 / n;
    double (*v)[n-1] = (void*)v_;
    double r = 0.0;
    if (j > 0)
        r += v[i][j-1] * h;
    if (j < n-1)
        r -= v[i][j] * h;
    return r;
}

void add_b_y_t(int size, double* v_, double* r_i_, double m)
{
    int n = size;
    double (*r_i)[n] = (void*)r_i_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            r_i[i][j] += m * calc_b_y_t_step(n, v_, i, j);
    return ;
}

double calc_res_i_norm(int size, double* u_, double* v_, double* c_i_)
{
    int n = size;
    double (*c_i)[n] = (void*)c_i_;
    double r_i_2 = 0.0;
#pragma omp parallel for collapse(2) reduction(+: r_i_2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            double r = c_i[i][j] - calc_b_x_t_step(n, u_, i, j) - calc_b_y_t_step(n, v_, i, j);
            r_i_2 += r*r;
        }
    return sqrt(r_i_2);
}

inline double calc_a_x_step(int size, double* u_, int i, int j)
{
    int n = size;
    double (*u)[n] = (void*)u_;
    double r = 0.0;
    if (j == 0)
        r = 3.0 * u[i][j] - u[i][j+1];
    else if (j == n-1)
        r = 3.0 * u[i][j] - u[i][j-1];
    else
        r = 4.0 * u[i][j] - u[i][j-1] - u[i][j+1];
    if (i < n-2)
        r -= u[i+1][j];
    if (i > 0)
        r -= u[i-1][j];
    return r;
}

void add_a_x(int size, double* u_, double* r_x_, double m)
{
    int n = size;
    double (*r_x)[n] = (void*)r_x_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            r_x[i][j] += m * calc_a_x_step(n, u_, i, j);
    return ;
}

// void add_a_x(int size, double* u_, double* r_x_, double m)
// {
//     int n = size;
//     double (*u)[n] = (void*)u_, (*r_x)[n] = (void*)r_x_;
//     cblas_daxpy(n-1, 3.0*m, &u[0][0], n, &r_x[0][0], n);
//     cblas_daxpy(n-1, 3.0*m, &u[0][n-1], n, &r_x[0][n-1], n);
//     for (int i = 0; i < n-1; i++)
//         cblas_daxpy(n-2, 4.0*m, &u[i][1], 1, &r_x[i][1], 1);
//     for (int i = 0; i < n-2; i++)
//     {
//         cblas_daxpy(n, -m, &u[i][0], 1, &r_x[i+1][0], 1);
//         cblas_daxpy(n, -m, &u[i+1][0], 1, &r_x[i][0], 1);
//     }
//     for (int i = 0; i < n-1; i++)
//     {
//         cblas_daxpy(n-1, -m, &u[i][0], 1, &r_x[i][1], 1);
//         cblas_daxpy(n-1, -m, &u[i][1], 1, &r_x[i][0], 1);
//     }
//     return ;
// }

inline double calc_b_x_step(int size, double* p_, int i, int j)
{
    int n = size;
    double h = 1.0 / n;
    double (*p)[n] = (void*)p_;
    double r = h * (p[i+1][j] - p[i][j]);
    return r;
}

void add_b_x(int size, double* p_, double* r_x_, double m)
{
    int n = size;
    double (*r_x)[n] = (void*)r_x_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
            r_x[i][j] += m * calc_b_x_step(n, p_, i, j);
    return ;
}

double calc_res_x_norm(int size, double* u_, double* p_, double* c_x_)
{
    int n = size;
    double r_x_2 = 0.0;
    double (*c_x)[n] = (void*)c_x_;
#pragma omp parallel for collapse(2) reduction(+: r_x_2)
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n; j++)
        {
            double r = c_x[i][j] - calc_a_x_step(n, u_, i, j) - calc_b_x_step(n, p_, i, j);
            r_x_2 += r*r;
        }
    return sqrt(r_x_2);
}

inline double calc_a_y_step(int size, double* v_, int i, int j)
{
    int n = size;
    double (*v)[n-1] = (void*)v_;
    double r = 0.0;
    if (i == 0)
        r = 3.0 * v[i][j] - v[i+1][j];
    else if (i == n-1)
        r = 3.0 * v[i][j] - v[i-1][j];
    else
        r = 4.0 * v[i][j] - v[i-1][j] - v[i+1][j];
    if (j < n-2)
        r -= v[i][j+1];
    if (j > 0)
        r -= v[i][j-1];
    return r;
}

void add_a_y(int size, double* v_, double* r_y_, double m)
{
    int n = size;
    double (*r_y)[n-1] = (void*)r_y_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            r_y[i][j] += m * calc_a_y_step(n, v_, i, j);
    return ;
}

inline double calc_b_y_step(int size, double* p_, int i, int j)
{
    int n = size;
    double h = 1.0 / n;
    double (*p)[n] = (void*)p_;
    double r = h * (p[i][j+1] - p[i][j]);
    return r;
}

void add_b_y(int size, double* p_, double* r_y_, double m)
{
    int n = size;
    double (*r_y)[n-1] = (void*)r_y_;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            r_y[i][j] += m * calc_b_y_step(n, p_, i, j);
    return ;
}

double calc_res_y_norm(int size, double* v_, double* p_, double* c_y_)
{
    int n = size;
    double r_y_2 = 0.0;
    double (*c_y)[n-1] = (void*)c_y_;
#pragma omp parallel for collapse(2) reduction(+: r_y_2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
        {
            double r = c_y[i][j] - calc_a_y_step(n, v_, i, j) - calc_b_y_step(n, p_, i, j);
            r_y_2 += r*r;
        }
    return sqrt(r_y_2);
}
