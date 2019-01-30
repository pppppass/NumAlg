#include "op.h"

static PyObject* wrapper_iter_dgs_incomp(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* u_obj, * v_obj, * p_obj, * c_i_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!O!O!i",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &c_i_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2),
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2),
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2),
        * c_i_arr = (PyArrayObject*)PyArray_FROM_OTF(c_i_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!u_arr || !v_arr || !p_arr || !c_i_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * v = PyArray_DATA(v_arr), * p = PyArray_DATA(p_arr), * c_i = PyArray_DATA(c_i_arr);

    omp_set_num_threads(num_threads);

    iter_dgs_incomp(n, u, v, p, c_i);

    Py_DECREF(c_i_arr);
    PyArray_ResolveWritebackIfCopy(p_arr);
    Py_DECREF(p_arr);
    PyArray_ResolveWritebackIfCopy(v_arr);
    Py_DECREF(v_arr);
    PyArray_ResolveWritebackIfCopy(u_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("");
}

static PyObject* wrapper_iter_gs_u(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* u_obj, * p_obj, * c_x_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!O!i",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &c_x_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2),
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * c_x_arr = (PyArrayObject*)PyArray_FROM_OTF(c_x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!u_arr || !p_arr || !c_x_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * p = PyArray_DATA(p_arr), * c_x = PyArray_DATA(c_x_arr);

    omp_set_num_threads(num_threads);

    iter_gs_u(n, u, p, c_x);

    Py_DECREF(c_x_arr);
    Py_DECREF(p_arr);
    PyArray_ResolveWritebackIfCopy(u_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("");
}

static PyObject* wrapper_iter_gs_v(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* v_obj, * p_obj, * c_y_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!O!i",
        &n,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &c_y_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2),
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * c_y_arr = (PyArrayObject*)PyArray_FROM_OTF(c_y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!v_arr || !p_arr || !c_y_arr)
        return NULL;

    double* v = PyArray_DATA(v_arr), * p = PyArray_DATA(p_arr), * c_y = PyArray_DATA(c_y_arr);

    omp_set_num_threads(num_threads);

    iter_gs_v(n, v, p, c_y);

    Py_DECREF(c_y_arr);
    Py_DECREF(p_arr);
    PyArray_ResolveWritebackIfCopy(v_arr);
    Py_DECREF(v_arr);

    return Py_BuildValue("");
}

static PyObject* wrapper_add_b_x_t(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* u_obj, * r_i_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &r_i_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * r_i_arr = (PyArrayObject*)PyArray_FROM_OTF(r_i_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!u_arr || !r_i_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * r_i = PyArray_DATA(r_i_arr);

    omp_set_num_threads(num_threads);

    add_b_x_t(n, u, r_i, m);

    PyArray_ResolveWritebackIfCopy(r_i_arr);
    Py_DECREF(r_i_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("O", r_i_obj);
}

static PyObject* wrapper_add_b_y_t(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* v_obj, * r_i_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &r_i_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * r_i_arr = (PyArrayObject*)PyArray_FROM_OTF(r_i_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!v_arr || !r_i_arr)
        return NULL;

    double* v = PyArray_DATA(v_arr), * r_i = PyArray_DATA(r_i_arr);

    omp_set_num_threads(num_threads);

    add_b_y_t(n, v, r_i, m);

    PyArray_ResolveWritebackIfCopy(r_i_arr);
    Py_DECREF(r_i_arr);
    Py_DECREF(v_arr);

    return Py_BuildValue("O", r_i_obj);
}

static PyObject* wrapper_calc_res_i_norm(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* u_obj, * v_obj, * c_i_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!O!i",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &c_i_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * c_i_arr = (PyArrayObject*)PyArray_FROM_OTF(c_i_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!u_arr || !v_arr || !c_i_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * v = PyArray_DATA(v_arr), * c_i = PyArray_DATA(c_i_arr);

    omp_set_num_threads(num_threads);

    double r = calc_res_i_norm(n, u, v, c_i);

    Py_DECREF(c_i_arr);
    Py_DECREF(v_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("d", r);
}

static PyObject* wrapper_add_a_x(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* u_obj, * r_x_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &r_x_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * r_x_arr = (PyArrayObject*)PyArray_FROM_OTF(r_x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!u_arr || !r_x_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * r_x = PyArray_DATA(r_x_arr);

    omp_set_num_threads(num_threads);

    add_a_x(n, u, r_x, m);

    PyArray_ResolveWritebackIfCopy(r_x_arr);
    Py_DECREF(r_x_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("O", r_x_arr);
}

static PyObject* wrapper_add_b_x(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* p_obj, * r_x_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &r_x_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * r_x_arr = (PyArrayObject*)PyArray_FROM_OTF(r_x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!p_arr || !r_x_arr)
        return NULL;

    double* p = PyArray_DATA(p_arr), * r_x = PyArray_DATA(r_x_arr);

    omp_set_num_threads(num_threads);

    add_b_x(n, p, r_x, m);

    PyArray_ResolveWritebackIfCopy(r_x_arr);
    Py_DECREF(r_x_arr);
    Py_DECREF(p_arr);

    return Py_BuildValue("O", r_x_arr);
}

static PyObject* wrapper_calc_res_x_norm(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* u_obj, * p_obj, * c_x_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!O!i",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &c_x_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * c_x_arr = (PyArrayObject*)PyArray_FROM_OTF(c_x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!u_arr || !p_arr || !c_x_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * p = PyArray_DATA(p_arr), * c_x = PyArray_DATA(c_x_arr);

    omp_set_num_threads(num_threads);

    double r = calc_res_x_norm(n, u, p, c_x);

    Py_DECREF(c_x_arr);
    Py_DECREF(p_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("d", r);
}

static PyObject* wrapper_add_a_y(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* v_obj, * r_y_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &r_y_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * r_y_arr = (PyArrayObject*)PyArray_FROM_OTF(r_y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!v_arr || !r_y_arr)
        return NULL;

    double* v = PyArray_DATA(v_arr), * r_y = PyArray_DATA(r_y_arr);

    omp_set_num_threads(num_threads);

    add_a_y(n, v, r_y, m);

    PyArray_ResolveWritebackIfCopy(r_y_arr);
    Py_DECREF(r_y_arr);
    Py_DECREF(v_arr);

    return Py_BuildValue("O", r_y_obj);
}

static PyObject* wrapper_add_b_y(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* p_obj, * r_y_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &r_y_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * r_y_arr = (PyArrayObject*)PyArray_FROM_OTF(r_y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!p_arr || !r_y_arr)
        return NULL;

    double* p = PyArray_DATA(p_arr), * r_y = PyArray_DATA(r_y_arr);

    omp_set_num_threads(num_threads);

    add_b_y(n, p, r_y, m);

    PyArray_ResolveWritebackIfCopy(r_y_arr);
    Py_DECREF(r_y_arr);
    Py_DECREF(p_arr);

    return Py_BuildValue("O", r_y_obj);
}

static PyObject* wrapper_calc_res_y_norm(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* v_obj, * p_obj, * c_y_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!O!i",
        &n,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &c_y_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * c_y_arr = (PyArrayObject*)PyArray_FROM_OTF(c_y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!v_arr || !p_arr || !c_y_arr)
        return NULL;

    double* v = PyArray_DATA(v_arr), * p = PyArray_DATA(p_arr), * c_y = PyArray_DATA(c_y_arr);

    omp_set_num_threads(num_threads);

    double r = calc_res_y_norm(n, v, p, c_y);

    Py_DECREF(c_y_arr);
    Py_DECREF(p_arr);
    Py_DECREF(v_arr);

    return Py_BuildValue("d", r);
}

static PyObject* wrapper_add_pro_close_p(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* p_obj, * p_pro_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &p_pro_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * p_pro_arr = (PyArrayObject*)PyArray_FROM_OTF(p_pro_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!p_arr || !p_pro_arr)
        return NULL;

    double* p = PyArray_DATA(p_arr), * p_pro = PyArray_DATA(p_pro_arr);

    omp_set_num_threads(num_threads);

    add_pro_close_p(n, p, p_pro, m);

    PyArray_ResolveWritebackIfCopy(p_pro_arr);
    Py_DECREF(p_pro_arr);
    Py_DECREF(p_arr);

    return Py_BuildValue("O", p_pro_arr);
}

static PyObject* wrapper_add_pro_bil_u(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* u_obj, * u_pro_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &u_pro_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * u_pro_arr = (PyArrayObject*)PyArray_FROM_OTF(u_pro_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!u_arr || !u_pro_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * u_pro = PyArray_DATA(u_pro_arr);

    omp_set_num_threads(num_threads);

    add_pro_bil_u(n, u, u_pro, m);

    PyArray_ResolveWritebackIfCopy(u_pro_arr);
    Py_DECREF(u_pro_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("O", u_pro_arr);
}

static PyObject* wrapper_add_pro_bil_v(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* v_obj, * v_pro_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &v_pro_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * v_pro_arr = (PyArrayObject*)PyArray_FROM_OTF(v_pro_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!v_arr || !v_pro_arr)
        return NULL;

    double* v = PyArray_DATA(v_arr), * v_pro = PyArray_DATA(v_pro_arr);

    omp_set_num_threads(num_threads);

    add_pro_bil_v(n, v, v_pro, m);

    PyArray_ResolveWritebackIfCopy(v_pro_arr);
    Py_DECREF(v_pro_arr);
    Py_DECREF(v_arr);

    return Py_BuildValue("O", v_pro_arr);
}

static PyObject* wrapper_add_res_close_p(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* p_obj, * p_res_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &p_obj,
        &PyArray_Type, &p_res_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * p_arr = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * p_res_arr = (PyArrayObject*)PyArray_FROM_OTF(p_res_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!p_arr || !p_res_arr)
        return NULL;

    double* p = PyArray_DATA(p_arr), * p_res = PyArray_DATA(p_res_arr);

    omp_set_num_threads(num_threads);

    add_res_close_p(n, p, p_res, m);

    PyArray_ResolveWritebackIfCopy(p_res_arr);
    Py_DECREF(p_res_arr);
    Py_DECREF(p_arr);

    return Py_BuildValue("O", p_res_arr);
}

static PyObject* wrapper_add_res_close_u(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* u_obj, * u_res_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &u_obj,
        &PyArray_Type, &u_res_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * u_res_arr = (PyArrayObject*)PyArray_FROM_OTF(u_res_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!u_arr || !u_res_arr)
        return NULL;

    double* u = PyArray_DATA(u_arr), * u_res = PyArray_DATA(u_res_arr);

    omp_set_num_threads(num_threads);

    add_res_close_u(n, u, u_res, m);

    PyArray_ResolveWritebackIfCopy(u_res_arr);
    Py_DECREF(u_res_arr);
    Py_DECREF(u_arr);

    return Py_BuildValue("O", u_res_arr);
}

static PyObject* wrapper_add_res_close_v(PyObject* self, PyObject* args)
{
    int n, num_threads;
    double m;
    PyObject* v_obj, * v_res_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!di",
        &n,
        &PyArray_Type, &v_obj,
        &PyArray_Type, &v_res_obj,
        &m,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * v_res_arr = (PyArrayObject*)PyArray_FROM_OTF(v_res_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!v_arr || !v_res_arr)
        return NULL;

    double* v = PyArray_DATA(v_arr), * v_res = PyArray_DATA(v_res_arr);

    omp_set_num_threads(num_threads);

    add_res_close_v(n, v, v_res, m);

    PyArray_ResolveWritebackIfCopy(v_res_arr);
    Py_DECREF(v_res_arr);
    Py_DECREF(v_arr);

    return Py_BuildValue("O", v_res_arr);
}

static PyObject* wrapper_sol_a_x_spec(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* a_x_u_obj, * u_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!i",
        &n,
        &PyArray_Type, &a_x_u_obj,
        &PyArray_Type, &u_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_x_u_arr = (PyArrayObject*)PyArray_FROM_OTF(a_x_u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_x_u_arr || !u_arr)
        return NULL;

    double* a_x_u = PyArray_DATA(a_x_u_arr), * u = PyArray_DATA(u_arr);

    omp_set_num_threads(num_threads);
    
    fftw_init_threads();
    fftw_plan_with_nthreads(num_threads);

    double* w = fftw_malloc((n-1)*n * sizeof(double));

    sol_a_x_spec(n, a_x_u, u, w);

    fftw_free(w);

    fftw_cleanup_threads();

    PyArray_ResolveWritebackIfCopy(u_arr);
    Py_DECREF(u_arr);
    Py_DECREF(a_x_u_arr);

    return Py_BuildValue("O", u_arr);
}

static PyObject* wrapper_sol_a_y_spec(PyObject* self, PyObject* args)
{
    int n, num_threads;
    PyObject* a_y_v_obj, * v_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!i",
        &n,
        &PyArray_Type, &a_y_v_obj,
        &PyArray_Type, &v_obj,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_y_v_arr = (PyArrayObject*)PyArray_FROM_OTF(a_y_v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_y_v_arr || !v_arr)
        return NULL;

    double* a_y_v = PyArray_DATA(a_y_v_arr), * v = PyArray_DATA(v_arr);

    omp_set_num_threads(num_threads);

    fftw_init_threads();
    fftw_plan_with_nthreads(num_threads);

    double* w = fftw_malloc(n*(n-1) * sizeof(double));

    sol_a_y_spec(n, a_y_v, v, w);

    fftw_free(w);

    fftw_cleanup_threads();

    PyArray_ResolveWritebackIfCopy(v_arr);
    Py_DECREF(v_arr);
    Py_DECREF(a_y_v_arr);

    return Py_BuildValue("O", v_arr);
}

static PyObject* wrapper_sol_a_x_cg(PyObject* self, PyObject* args)
{
    int n, iter_max, num_threads;
    double eps;
    PyObject* a_x_u_obj, * u_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!dii",
        &n,
        &PyArray_Type, &a_x_u_obj,
        &PyArray_Type, &u_obj,
        &eps,
        &iter_max,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_x_u_arr = (PyArrayObject*)PyArray_FROM_OTF(a_x_u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_x_u_arr || !u_arr)
        return NULL;

    double* a_x_u = PyArray_DATA(a_x_u_arr), * u = PyArray_DATA(u_arr);

    omp_set_num_threads(num_threads);
    // mkl_set_num_threads(num_threads);

    double* w = malloc(3*(n-1)*n * sizeof(double));

    int ctr = sol_a_x_cg(n, a_x_u, u, w, eps, iter_max);

    free(w);

    PyArray_ResolveWritebackIfCopy(u_arr);
    Py_DECREF(u_arr);
    Py_DECREF(a_x_u_arr);

    return Py_BuildValue("iO", ctr, u_arr);
}

static PyObject* wrapper_sol_a_y_cg(PyObject* self, PyObject* args)
{
    int n, iter_max, num_threads;
    double eps;
    PyObject* a_y_v_obj, * v_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!dii",
        &n,
        &PyArray_Type, &a_y_v_obj,
        &PyArray_Type, &v_obj,
        &eps,
        &iter_max,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_y_v_arr = (PyArrayObject*)PyArray_FROM_OTF(a_y_v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_y_v_arr || !v_arr)
        return NULL;

    double* a_y_v = PyArray_DATA(a_y_v_arr), * v = PyArray_DATA(v_arr);

    omp_set_num_threads(num_threads);
    // mkl_set_num_threads(num_threads);


    double* w = malloc(3*n*(n-1) * sizeof(double));

    int ctr = sol_a_y_cg(n, a_y_v, v, w, eps, iter_max);

    free(w);

    PyArray_ResolveWritebackIfCopy(v_arr);
    Py_DECREF(v_arr);
    Py_DECREF(a_y_v_arr);

    return Py_BuildValue("iO", ctr, v_arr);
}

static PyObject* wrapper_iter_a_x_mg_gs(PyObject* self, PyObject* args)
{
    int n, l, nu_down, nu_up, num_threads;
    PyObject* a_x_u_obj, * u_obj;

    if(!PyArg_ParseTuple(
        args, "iiO!O!iii",
        &n, &l,
        &PyArray_Type, &a_x_u_obj,
        &PyArray_Type, &u_obj,
        &nu_down,
        &nu_up,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_x_u_arr = (PyArrayObject*)PyArray_FROM_OTF(a_x_u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_x_u_arr || !u_arr)
        return NULL;

    double* a_x_u = PyArray_DATA(a_x_u_arr), * u = PyArray_DATA(u_arr);

    omp_set_num_threads(num_threads);
    // mkl_set_num_threads(num_threads);

    int size_w = 0;
    {
        int n_temp = n;
        for (int i = 0; i < l; i++)
        {
            size_w += (n_temp-1)*n_temp;
            n_temp /= 2;
            size_w += (n_temp-1)*n_temp;
        }
    }
    double* w = malloc(size_w * sizeof(double));

    iter_sol_a_x_mg_gs(n, l, a_x_u, u, w, nu_down, nu_up);

    free(w);

    PyArray_ResolveWritebackIfCopy(u_arr);
    Py_DECREF(u_arr);
    Py_DECREF(a_x_u_arr);

    return Py_BuildValue("");
}

static PyObject* wrapper_iter_a_y_mg_gs(PyObject* self, PyObject* args)
{
    int n, l, nu_down, nu_up, num_threads;
    PyObject* a_y_v_obj, * v_obj;

    if(!PyArg_ParseTuple(
        args, "iiO!O!iii",
        &n, &l,
        &PyArray_Type, &a_y_v_obj,
        &PyArray_Type, &v_obj,
        &nu_down,
        &nu_up,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_y_v_arr = (PyArrayObject*)PyArray_FROM_OTF(a_y_v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_y_v_arr || !v_arr)
        return NULL;

    double* a_y_v = PyArray_DATA(a_y_v_arr), * v = PyArray_DATA(v_arr);

    omp_set_num_threads(num_threads);
    // mkl_set_num_threads(num_threads);

    int size_w = 0;
    {
        int n_temp = n;
        for (int i = 0; i < l; i++)
        {
            size_w += n_temp*(n_temp-1);
            n_temp /= 2;
            size_w += n_temp*(n_temp-1);
        }
    }
    double* w = malloc(size_w * sizeof(double));

    iter_sol_a_y_mg_gs(n, l, a_y_v, v, w, nu_down, nu_up);

    free(w);

    PyArray_ResolveWritebackIfCopy(v_arr);
    Py_DECREF(v_arr);
    Py_DECREF(a_y_v_arr);

    return Py_BuildValue("");
}

static PyObject* wrapper_sol_a_x_pcg_mg_gs(PyObject* self, PyObject* args)
{
    int n, iter_max, l, nu_down, nu_up, num_threads;
    double eps;
    PyObject* a_x_u_obj, * u_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!diiiii",
        &n,
        &PyArray_Type, &a_x_u_obj,
        &PyArray_Type, &u_obj,
        &eps,
        &iter_max,
        &l, &nu_down, &nu_up,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_x_u_arr = (PyArrayObject*)PyArray_FROM_OTF(a_x_u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_x_u_arr || !u_arr)
        return NULL;

    double* a_x_u = PyArray_DATA(a_x_u_arr), * u = PyArray_DATA(u_arr);

    omp_set_num_threads(num_threads);
    // mkl_set_num_threads(num_threads);

    int size_w = 4*(n-1)*n;
    {
        int n_temp = n;
        for (int i = 0; i < l; i++)
        {
            size_w += (n_temp-1)*n_temp;
            n_temp /= 2;
            size_w += (n_temp-1)*n_temp;
        }
    }
    double* w = malloc(size_w * sizeof(double));

    int ctr = sol_a_x_pcg_mg_gs(n, a_x_u, u, w, eps, iter_max, l, nu_down, nu_up);

    free(w);

    PyArray_ResolveWritebackIfCopy(u_arr);
    Py_DECREF(u_arr);
    Py_DECREF(a_x_u_arr);

    return Py_BuildValue("iO", ctr, u_arr);
}

static PyObject* wrapper_sol_a_y_pcg_mg_gs(PyObject* self, PyObject* args)
{
    int n, iter_max, l, nu_down, nu_up, num_threads;
    double eps;
    PyObject* a_y_v_obj, * v_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!diiiii",
        &n,
        &PyArray_Type, &a_y_v_obj,
        &PyArray_Type, &v_obj,
        &eps,
        &iter_max,
        &l, &nu_down, &nu_up,
        &num_threads
    ))
        return NULL;
    
    PyArrayObject
        * a_y_v_arr = (PyArrayObject*)PyArray_FROM_OTF(a_y_v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        * v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!a_y_v_arr || !v_arr)
        return NULL;

    double* a_y_v = PyArray_DATA(a_y_v_arr), * v = PyArray_DATA(v_arr);

    omp_set_num_threads(num_threads);
    // mkl_set_num_threads(num_threads);

    int size_w = 4*n*(n-1);
    {
        int n_temp = n;
        for (int i = 0; i < l; i++)
        {
            size_w += n_temp*(n_temp-1);
            n_temp /= 2;
            size_w += n_temp*(n_temp-1);
        }
    }
    double* w = malloc(size_w * sizeof(double));

    int ctr = sol_a_y_pcg_mg_gs(n, a_y_v, v, w, eps, iter_max, l, nu_down, nu_up);

    free(w);

    PyArray_ResolveWritebackIfCopy(v_arr);
    Py_DECREF(v_arr);
    Py_DECREF(a_y_v_arr);

    return Py_BuildValue("iO", ctr, v_arr);
}

static PyMethodDef methods[] =
{
    {"wrapper_iter_dgs_incomp", wrapper_iter_dgs_incomp, METH_VARARGS, NULL},
    {"wrapper_iter_gs_u", wrapper_iter_gs_u, METH_VARARGS, NULL},
    {"wrapper_iter_gs_v", wrapper_iter_gs_v, METH_VARARGS, NULL},
    {"wrapper_add_b_x_t", wrapper_add_b_x_t, METH_VARARGS, NULL},
    {"wrapper_add_b_y_t", wrapper_add_b_y_t, METH_VARARGS, NULL},
    {"wrapper_calc_res_i_norm", wrapper_calc_res_i_norm, METH_VARARGS, NULL},
    {"wrapper_add_a_x", wrapper_add_a_x, METH_VARARGS, NULL},
    {"wrapper_add_b_x", wrapper_add_b_x, METH_VARARGS, NULL},
    {"wrapper_calc_res_x_norm", wrapper_calc_res_x_norm, METH_VARARGS, NULL},
    {"wrapper_add_a_y", wrapper_add_a_y, METH_VARARGS, NULL},
    {"wrapper_add_b_y", wrapper_add_b_y, METH_VARARGS, NULL},
    {"wrapper_calc_res_y_norm", wrapper_calc_res_y_norm, METH_VARARGS, NULL},
    {"wrapper_add_pro_close_p", wrapper_add_pro_close_p, METH_VARARGS, NULL},
    {"wrapper_add_pro_bil_u", wrapper_add_pro_bil_u, METH_VARARGS, NULL},
    {"wrapper_add_pro_bil_v", wrapper_add_pro_bil_v, METH_VARARGS, NULL},
    {"wrapper_add_res_close_p", wrapper_add_res_close_p, METH_VARARGS, NULL},
    {"wrapper_add_res_close_u", wrapper_add_res_close_u, METH_VARARGS, NULL},
    {"wrapper_add_res_close_v", wrapper_add_res_close_v, METH_VARARGS, NULL},
    {"wrapper_sol_a_x_spec", wrapper_sol_a_x_spec, METH_VARARGS, NULL},
    {"wrapper_sol_a_y_spec", wrapper_sol_a_y_spec, METH_VARARGS, NULL},
    {"wrapper_sol_a_x_cg", wrapper_sol_a_x_cg, METH_VARARGS, NULL},
    {"wrapper_sol_a_y_cg", wrapper_sol_a_y_cg, METH_VARARGS, NULL},
    {"wrapper_iter_a_x_mg_gs", wrapper_iter_a_x_mg_gs, METH_VARARGS, NULL},
    {"wrapper_iter_a_y_mg_gs", wrapper_iter_a_y_mg_gs, METH_VARARGS, NULL},
    {"wrapper_sol_a_x_pcg_mg_gs", wrapper_sol_a_x_pcg_mg_gs, METH_VARARGS, NULL},
    {"wrapper_sol_a_y_pcg_mg_gs", wrapper_sol_a_y_pcg_mg_gs, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "op", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_op(void)
{
    import_array();
    return PyModule_Create(&table);
}
