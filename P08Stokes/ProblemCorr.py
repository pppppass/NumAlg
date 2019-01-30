
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


# In[14]:


n = 256
eps_uzawa = 1.0e-10
taus = [0] + [10.0**(k / 2) for k in [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]]
eps_cg = 1.0e-11
iter_cg = 5000
alphas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
nu_down, nu_up = 0, 1
iter_max = 100
l = 0
k = 4
rt = []


# In[15]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[16]:


for tau in taus:
    for alpha in alphas:
        u = numpy.zeros((n-1, n))
        v = numpy.zeros((n, n-1))
        p = numpy.zeros((n, n))
        u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_cg_mg(n, alpha, eps_cg, iter_cg, tau, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
        res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
        err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
        print("tau = {}, alpha = {} finished".format(tau, alpha))
        rt.append((tau, alpha, ctr, elapsed, res, err))


# In[ ]:


db["Prob3/Corr/VarTauAlpha"] = rt
db.sync()


# In[30]:


n = 256
eps_uzawa = 1.0e-10
taus = [0.0, 0.01, 0.1, 1.0, 10.0]
eps_cg = 1.0e-11
iter_cg = 5000
alpha = 1.0
nu_down, nu_up = 2, 2
iter_max = 100
ls = [0, 1, 2, 3, 4, 5, 6]
k = 4
rt = []


# In[31]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[32]:


for l in ls:
    for tau in taus:
        u = numpy.zeros((n-1, n))
        v = numpy.zeros((n, n-1))
        p = numpy.zeros((n, n))
        u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_cg_mg(n, alpha, eps_cg, iter_cg, tau, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
        res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
        err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
        print("l = {}, tau = {} finished".format(l, tau))
        rt.append((l, tau, ctr, elapsed, res, err))


# In[ ]:


db["Prob3/Corr/VarTauL/Nu22"] = rt
db.sync()


# In[33]:


n = 256
eps_uzawa = 1.0e-10
taus = [0.0, 0.01, 0.1, 1.0, 10.0]
eps_cg = 1.0e-11
iter_cg = 5000
alpha = 1.0
nu_down, nu_up = 0, 1
iter_max = 100
ls = [0, 1, 2, 3, 4, 5, 6]
k = 4
rt = []


# In[34]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[35]:


for l in ls:
    for tau in taus:
        u = numpy.zeros((n-1, n))
        v = numpy.zeros((n, n-1))
        p = numpy.zeros((n, n))
        u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_cg_mg(n, alpha, eps_cg, iter_cg, tau, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
        res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
        err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
        print("l = {}, tau = {} finished".format(l, tau))
        rt.append((l, tau, ctr, elapsed, res, err))


# In[ ]:


db["Prob3/Corr/VarTauL/Nu01"] = rt
db.sync()


# In[3]:


n = 256
eps_uzawa = 1.0e-10
l_pcg = 6
nu_down_pcg, nu_up_pcg = 4, 4
taus = [0.0, 0.01, 0.1, 1.0, 10.0]
eps_cg = 1.0e-11
iter_cg = 5000
alpha = 1.0
nu_down, nu_up = 2, 2
iter_max = 100
ls = [0, 1, 2, 3, 4, 5, 6]
k = 4
rt = []


# In[4]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[5]:


for l in ls:
    for tau in taus:
        u = numpy.zeros((n-1, n))
        v = numpy.zeros((n, n-1))
        p = numpy.zeros((n, n))
        u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_pcg_mg_gs_mg(n, alpha, eps_cg, iter_cg, tau, l_pcg, nu_down_pcg, nu_up_pcg, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
        res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
        err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
        print("l = {}, tau = {} finished".format(l, tau))
        rt.append((l, tau, ctr, elapsed, res, err))


# In[ ]:


db["Prob4/Corr/VarTauL/LPCG6"] = rt
db.sync()


# In[15]:


n = 256
eps_uzawa = 1.0e-10
l_pcg = 5
nu_down_pcg, nu_up_pcg = 4, 4
taus = [0.0, 0.01, 0.1, 1.0, 10.0]
eps_cg = 1.0e-11
iter_cg = 5000
alpha = 1.0
nu_down, nu_up = 2, 2
iter_max = 100
ls = [0, 1, 2, 3, 4, 5, 6]
k = 4
rt = []


# In[16]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[17]:


for l in ls:
    for tau in taus:
        u = numpy.zeros((n-1, n))
        v = numpy.zeros((n, n-1))
        p = numpy.zeros((n, n))
        u, v, p, ctr, elapsed = drivers.driver_uzawa_inexact_pcg_mg_gs_mg(n, alpha, eps_cg, iter_cg, tau, l_pcg, nu_down_pcg, nu_up_pcg, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
        res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
        err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
        print("l = {}, tau = {} finished".format(l, tau))
        rt.append((l, tau, ctr, elapsed, res, err))


# In[ ]:


db["Prob4/Corr/VarTauL/LPCG5"] = rt
db.sync()


# In[ ]:


db.close()

