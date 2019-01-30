#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
// #include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <fftw3.h>

#define SIZE_MIN_PARALLEL 16

void iter_dgs_incomp(int size, double* u_, double* v_, double* p_, double* c_i_);
void iter_gs_u(int size, double* u_, double* p_, double* c_x_);
void iter_gs_v(int size, double* v_, double* p_, double* c_y_);

// Residue
void add_b_x_t(int size, double* u_, double* r_i_, double m);
void add_b_y_t(int size, double* v_, double* r_i_, double m);
double calc_res_i_norm(int size, double* u_, double* v_, double* c_i_);
void add_a_x(int size, double* u_, double* r_x_, double m);
void add_b_x(int size, double* p_, double* r_x_, double m);
double calc_res_x_norm(int size, double* u_, double* p_, double* c_x_);
void add_a_y(int size, double* v_, double* r_y_, double m);
void add_b_y(int size, double* p_, double* r_y_, double m);
double calc_res_y_norm(int size, double* v_, double* p_, double* c_y_);

void add_pro_close_p(int size, double* p_, double* p_pro_, double m);
void add_pro_bil_u(int size, double* u_, double* u_pro_, double m);
void add_pro_bil_v(int size, double* v_, double* v_pro_, double m);

// Restriction
void add_res_close_p(int size, double* p_, double* p_res_, double m);
void add_res_close_u(int size, double* u_, double* u_res_, double m);
void add_res_close_v(int size, double* v_, double* v_res_, double m);

void sol_a_x_spec(int size, double* a_x_u_, double* u_, double* w_);
void sol_a_y_spec(int size, double* a_y_v_, double* v_, double* w_);

int sol_a_x_cg(int size, double* a_x_u_, double* u_, double* w_, double eps, int iter_max);
int sol_a_y_cg(int size, double* a_y_v_, double* v_, double* w_, double eps, int iter_max);

void iter_sol_a_x_gs(int size, double* a_x_u_, double* u_);
void iter_sol_a_y_gs(int size, double* a_y_v_, double* v_);

void iter_sol_a_x_mg_gs(int size, int level, double* a_x_u_, double* u_, double* w_, int nu_down, int nu_up);
void iter_sol_a_y_mg_gs(int size, int level, double* a_y_v_, double* v_, double* w_, int nu_down, int nu_up);

int sol_a_x_pcg_mg_gs(int size, double* a_x_u_, double* u_, double* w_, double eps, int iter_max, int level, int nu_down, int nu_up);
int sol_a_y_pcg_mg_gs(int size, double* a_y_v_, double* v_, double* w_, double eps, int iter_max, int level, int nu_down, int nu_up);
