#include "op.h"

void inline iter_dgs_incomp_step(int size, double* u_, double* v_, double* p_, double* c_i_, int i, int j)
{
    int n = size;
    double h = 1.0 / n;
    double (*u)[n] = (void*)u_, (*v)[n-1] = (void*)v_, (*p)[n] = (void*)p_, (*c_i)[n] = (void*)c_i_;
    double dq;
    if (i == 0)
    {
        if (j == 0)
        {
            dq = (c_i[0][0] + (u[0][0] + v[0][0]) * h) / 2.0;
            u[0][0] -= dq / h;
            v[0][0] -= dq / h;
            p[0][0] -= 4.0 * dq / h / h;
            p[1][0] += dq / h / h;
            p[0][1] += dq / h / h;
        }
        else if (j == n-1)
        {
            dq = (c_i[0][n-1] + (u[0][n-1] - v[0][n-2]) * h) / 2.0;
            u[0][n-1] -= dq / h;
            v[0][n-2] += dq / h;
            p[0][n-1] -= 4.0 * dq / h / h;
            p[1][n-1] += dq / h / h;
            p[0][n-2] += dq / h / h;
        }
        else
        {
            dq = (c_i[0][j] + (u[0][j] + v[0][j] - v[0][j-1]) * h) / 3.0;
            u[0][j] -= dq / h;
            v[0][j] -= dq / h;
            v[0][j-1] += dq / h;
            p[0][j] -= 4.0 * dq / h / h;
            p[1][j] += dq / h / h;
            p[0][j-1] += dq / h / h;
            p[0][j+1] += dq / h / h;
        }
    }
    else if (i == n-1)
    {
        if (j == 0)
        {
            dq = (c_i[n-1][0] + (-u[n-2][0] + v[n-1][0]) * h) / 2.0;
            u[n-2][0] += dq / h;
            v[n-1][0] -= dq / h;
            p[n-1][0] -= 4.0 * dq / h / h;
            p[n-2][0] += dq / h / h;
            p[n-1][1] += dq / h / h;
        }
        else if (j == n-1)
        {
            dq = (c_i[n-1][n-1] + (-u[n-2][n-1] - v[n-1][n-2]) * h) / 2.0;
            u[n-2][n-1] += dq / h;
            v[n-1][n-2] += dq / h;
            p[n-1][n-1] -= 4.0 * dq / h / h;
            p[n-2][n-1] += dq / h / h;
            p[n-1][n-2] += dq / h / h;
        }
        else
        {
            dq = (c_i[n-1][j] + (-u[n-2][j] + v[n-1][j] - v[n-1][j-1]) * h) / 3.0;
            u[n-2][j] += dq / h;
            v[n-1][j] -= dq / h;
            v[n-1][j-1] += dq / h;
            p[n-1][j] -= 4.0 * dq / h / h;
            p[n-2][j] += dq / h / h;
            p[n-1][j-1] += dq / h / h;
            p[n-1][j+1] += dq / h / h;
        }
    }
    else
    {
        if (j == 0)
        {
            dq = (c_i[i][0] + (u[i][0] - u[i-1][0] + v[i][0]) * h) / 3.0;
            u[i][0] -= dq / h;
            u[i-1][0] += dq / h;
            v[i][0] -= dq / h;
            p[i][0] -= 4.0 * dq / h / h;
            p[i-1][0] += dq / h / h;
            p[i+1][0] += dq / h / h;
            p[i][1] += dq / h / h;
        }
        else if (j == n-1)
        {
            dq = (c_i[i][n-1] + (u[i][n-1] - u[i-1][n-1] - v[i][n-2]) * h) / 3.0;
            u[i][n-1] -= dq / h;
            u[i-1][n-1] += dq / h;
            v[i][n-2] += dq / h;
            p[i][n-1] -= 4.0 * dq / h / h;
            p[i-1][n-1] += dq / h / h;
            p[i+1][n-1] += dq / h / h;
            p[i][n-2] += dq / h / h;
        }
        else
        {
            dq = (c_i[i][j] + (u[i][j] - u[i-1][j] + v[i][j] - v[i][j-1]) * h) / 4.0;
            u[i][j] -= dq / h;
            u[i-1][j] += dq / h;
            v[i][j] -= dq / h;
            v[i][j-1] += dq / h;
            p[i][j] -= 4.0 * dq / h / h;
            p[i-1][j] += dq / h / h;
            p[i+1][j] += dq / h / h;
            p[i][j-1] += dq / h / h;
            p[i][j+1] += dq / h / h;
        }
    }
    return ;
}

void iter_dgs_incomp(int size, double* u_, double* v_, double* p_, double* c_i_)
{
    int n = size;
// #pragma omp parallel
//     {
//         int r = omp_get_thread_num(), k = omp_get_num_threads();
//         for (int i = 0; i < n; i++)
//             for (int j = 0; j < n; j++)
//                 iter_dgs_incomp_step(n, u_, v_, p_, c_i_, (i+r*n/k)%n, j);
//     }
#pragma omp parallel for collapse(2) if(n >= SIZE_MIN_PARALLEL)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            iter_dgs_incomp_step(n, u_, v_, p_, c_i_, i, j);
    return ;
}
