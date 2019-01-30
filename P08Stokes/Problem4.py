
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import models
import drivers
import op


# In[2]:


db = shelve.open("Result")


# In[3]:


ns = [256, 512, 1024, 2048, 4096, 8192]
ls = [6, 7, 8, 9, 10, 11]
nu_down, nu_up = 4, 4
eps_cg = 1.0e-10
iter_max = 100
k = 4
rt = []


# In[ ]:


for n, l in zip(ns, ls):
    numpy.random.seed(1)
    u_ana = numpy.random.randn(n-1, n)
    a_x_u_ana = op.wrapper_add_a_x(n, u_ana, numpy.zeros((n-1, n)), 1.0, k)
    start = time.time()
    ctr, u_sol = op.wrapper_sol_a_x_pcg_mg_gs(n, a_x_u_ana, numpy.zeros((n-1, n)), eps_cg, iter_max, l, nu_down, nu_up, k)
    end = time.time()
    a_x_u_sol = op.wrapper_add_a_x(n, u_sol, numpy.zeros((n-1, n)), 1.0, k)
    res = numpy.linalg.norm((a_x_u_ana - a_x_u_sol).flat)
    err = numpy.linalg.norm((u_ana - u_sol).flat)
    elapsed = end - start
    print("Summary:")
    print("\tRes.: {}".format(res))
    print("\tError: {}".format(err))
    print("\tIter: {}".format(ctr))
    print("\tElapsed: {}".format(elapsed))
    rt.append((n, res, err, ctr, elapsed))
    print("n = {} finished".format(n))
    del u_ana, a_x_u_ana, u_sol, a_x_u_sol


# In[ ]:


db["Prob3/Ell/PCG-MG/U"] = rt
db.sync()


# In[5]:


ns = [256, 512, 1024, 2048, 4096, 8192]
ls = [6, 7, 8, 9, 10, 11]
nu_down, nu_up = 4, 4
eps_cg = 1.0e-10
iter_max = 100
k = 4
rt = []


# In[6]:


for n, l in zip(ns, ls):
    numpy.random.seed(1)
    v_ana = numpy.random.randn(n-1, n)
    a_y_v_ana = op.wrapper_add_a_y(n, v_ana, numpy.zeros((n-1, n)), 1.0, k)
    start = time.time()
    ctr, v_sol = op.wrapper_sol_a_y_pcg_mg_gs(n, a_y_v_ana, numpy.zeros((n-1, n)), eps_cg, iter_max, l, nu_down, nu_up, k)
    end = time.time()
    a_y_v_sol = op.wrapper_add_a_y(n, v_sol, numpy.zeros((n-1, n)), 1.0, k)
    res = numpy.linalg.norm((a_y_v_ana - a_y_v_sol).flat)
    err = numpy.linalg.norm((v_ana - v_sol).flat)
    elapsed = end - start
    print("Summary:")
    print("\tRes.: {}".format(res))
    print("\tError: {}".format(err))
    print("\tIter: {}".format(ctr))
    print("\tElapsed: {}".format(elapsed))
    rt.append((n, res, err, ctr, elapsed))
    print("n = {} finished".format(n))
    del v_ana, a_y_v_ana, v_sol, a_y_v_sol


# In[ ]:


db["Prob3/Ell/PCG-MG/V"] = rt
db.sync()


# In[3]:


n = 256
ls = [0, 1, 2, 3, 4, 5, 6]
nu_down, nu_up = 4, 4
eps_cg = 1.0e-10
iter_max = 1000
k = 4
rt = []


# In[4]:


for l in ls:
    numpy.random.seed(1)
    u_ana = numpy.random.randn(n-1, n)
    a_x_u_ana = op.wrapper_add_a_x(n, u_ana, numpy.zeros((n-1, n)), 1.0, k)
    start = time.time()
    ctr, u_sol = op.wrapper_sol_a_x_pcg_mg_gs(n, a_x_u_ana, numpy.zeros((n-1, n)), eps_cg, iter_max, l, nu_down, nu_up, k)
    end = time.time()
    a_x_u_sol = op.wrapper_add_a_x(n, u_sol, numpy.zeros((n-1, n)), 1.0, k)
    res = numpy.linalg.norm((a_x_u_ana - a_x_u_sol).flat)
    err = numpy.linalg.norm((u_ana - u_sol).flat)
    elapsed = end - start
    print("Summary:")
    print("\tRes.: {}".format(res))
    print("\tError: {}".format(err))
    print("\tIter: {}".format(ctr))
    print("\tElapsed: {}".format(elapsed))
    rt.append((n, res, err, ctr, elapsed))
    print("l = {} finished".format(l))
    del u_ana, a_x_u_ana, u_sol, a_x_u_sol


# In[5]:


db["Prob3/Ell/PCG-MG/VarL"] = rt
db.sync()


# In[6]:


n = 256
eps_uzawa = 1.0e-10
eps_cgs = numpy.logspace(-6.0, -12.0, 13)
iter_cg = 5000
l_pcg = 6
nu_down_pcg, nu_up_pcg = 4, 4
alpha = 1.0
nu_down, nu_up = 0, 1
iter_max = 100
l = 0
k = 4
rt = []


# In[7]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[8]:


for eps_cg in eps_cgs:
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_pcg_mg_gs_mg(n, alpha, eps_cg, iter_cg, l_pcg, nu_down_pcg, nu_up_pcg, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("eps_cg = {} finished".format(eps_cg))
    rt.append((eps_cg, ctr, elapsed, res, err))


# In[9]:


db["Prob4/VarEpsCG"] = rt
db.sync()


# In[11]:


ns = [64, 128, 256, 512, 1024, 2048]#, 4096, 8192]
eps_uzawa = 1.0e-10
eps_cg = 1.0e-11
iter_cg = 30
l_pcgs = [4, 5, 6, 7, 8, 9, 10, 11]
nu_down_pcg, nu_up_pcg = 4, 4
alpha = 1.0
nu_down, nu_up = 0, 1
iter_max = 30
l = 0
k = 4
rt = []


# In[12]:


for n, l_pcg in zip(ns, l_pcgs):
    u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
    c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_pcg_mg_gs_mg(n, alpha, eps_cg, iter_cg, l_pcg, nu_down_pcg, nu_up_pcg, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("n = {} finished".format(n))
    rt.append((n, l, ctr, elapsed, res, err))


# In[13]:


db["Prob4/VarN/Prob1"] = rt
db.sync()


# In[14]:


ns = [64, 128, 256, 512, 1024, 2048]#, 4096, 8192]
eps_uzawa = 1.0e-10
eps_cg = 1.0e-11
iter_cg = 30
l_pcgs = [4, 5, 6, 7, 8, 9, 10, 11]
nu_down_pcg, nu_up_pcg = 4, 4
alpha = 1.0
nu_down, nu_up = 0, 1
iter_max = 30
l = 0
k = 4
rt = []


# In[15]:


for n, l_pcg in zip(ns, l_pcgs):
    u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_2(n)
    c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_pcg_mg_gs_mg(n, alpha, eps_cg, iter_cg, l_pcg, nu_down_pcg, nu_up_pcg, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("n = {} finished".format(n))
    rt.append((n, l, ctr, elapsed, res, err))


# In[16]:


db["Prob4/VarN/Prob2"] = rt
db.sync()


# In[ ]:


db.close()

