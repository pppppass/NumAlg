
# coding: utf-8

# In[1]:


import shelve
import numpy
import models
import drivers


# In[2]:


db = shelve.open("Result")


# In[3]:


n = 256
eps_dgs = 1.0e-10
nu_down, nu_up = 2, 2
iter_max = 5000
ls = [0, 1, 2, 3, 4, 5, 6]
k = 4
rt = []


# In[4]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[5]:


for l in ls:
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_dgs_mg(n, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_dgs, iter_max, False, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("L = {} finished".format(l))
    rt.append((l, ctr, elapsed, res, err))


# In[ ]:


db["Prob1/VarL"] = rt
db.sync()


# In[ ]:


n = 2048
eps_dgs = 1.0e-10
nus = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
iter_max = 100
l = 9
k = 4
rt = []


# In[ ]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[ ]:


for nu in nus:
    nu_down, nu_up = nu
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_dgs_mg(n, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_dgs, iter_max, False, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("nu = {} finished".format(nu))
    rt.append((nu, ctr, elapsed, res, err))


# In[ ]:


db["Prob1/VarNu"] = rt
db.sync()


# In[3]:


ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
eps_dgs = 1.0e-10
nu_down, nu_up = 2, 2
iter_max = 30
ls = [4, 5, 6, 7, 8, 9, 10, 11]
k = 4
rt = []


# In[4]:


for n, l in zip(ns, ls):
    u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
    c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_dgs_mg(n, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_dgs, iter_max, False, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("n = {} finished".format(n))
    rt.append((n, l, ctr, elapsed, res, err))


# In[5]:


db["Prob1/VarN/Prob1"] = rt
db.sync()


# In[6]:


ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
eps_dgs = 1.0e-10
nu_down, nu_up = 2, 2
iter_max = 30
ls = [4, 5, 6, 7, 8, 9, 10, 11]
k = 4
rt = []


# In[7]:


for n, l in zip(ns, ls):
    u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_2(n)
    c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_dgs_mg(n, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_dgs, iter_max, False, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("n = {} finished".format(n))
    rt.append((n, l, ctr, elapsed, res, err))


# In[ ]:


db["Prob1/VarN/Prob2"] = rt
db.sync()


# In[ ]:


db.close()

