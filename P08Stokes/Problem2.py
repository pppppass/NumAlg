
# coding: utf-8

# In[1]:


import op
import time
import shelve
import numpy
import models
import drivers
import spec


# In[2]:


db = shelve.open("Result")


# In[3]:


ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
k = 4
rt = []


# In[4]:


for n in ns:
    numpy.random.seed(1)
    u_ana = numpy.random.randn(n-1, n)
    a_x_u_ana = op.wrapper_add_a_x(n, u_ana, numpy.zeros((n-1, n)), 1.0, k)
    start = time.time()
#     u_sol = spec.sol_a_x_spec(n, a_x_u_ana)
    u_sol = op.wrapper_sol_a_x_spec(n, a_x_u_ana, numpy.zeros((n-1, n)), k)
    end = time.time()
    a_x_u_sol = op.wrapper_add_a_x(n, u_sol, numpy.zeros((n-1, n)), 1.0, k)
    res = numpy.linalg.norm((a_x_u_ana - a_x_u_sol).flat)
    err = numpy.linalg.norm((u_ana - u_sol).flat)
    elapsed = end - start
    print("Summary:")
    print("\tRes.: {}".format(res))
    print("\tError: {}".format(err))
    print("\tElapsed: {}".format(elapsed))
    rt.append((n, res, err, elapsed))
    print("n = {} finished".format(n))
    del u_ana, a_x_u_ana, u_sol, a_x_u_sol


# In[5]:


db["Prob2/Ell/Spec/U"] = rt
db.sync()


# In[6]:


ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
k = 4
rt = []


# In[7]:


for n in ns:
    numpy.random.seed(1)
    v_ana = numpy.random.randn(n, n-1)
    a_y_v_ana = op.wrapper_add_a_y(n, v_ana, numpy.zeros((n, n-1)), 1.0, k)
    start = time.time()
#     v_sol = spec.sol_a_y_spec(n, a_y_v_ana)
    v_sol = op.wrapper_sol_a_y_spec(n, a_y_v_ana, numpy.ones((n, n-1)), k)
    end = time.time()
    a_y_v_sol = op.wrapper_add_a_y(n, v_sol, numpy.zeros((n, n-1)), 1.0, k)
    res = numpy.linalg.norm((a_y_v_ana - a_y_v_sol).flat)
    err = numpy.linalg.norm((v_ana - v_sol).flat)
    elapsed = end - start
    print("Summary:")
    print("\tRes.: {}".format(res))
    print("\tError: {}".format(err))
    print("\tElapsed: {}".format(elapsed))
    rt.append((n, res, err, elapsed))
    print("n = {} finished".format(n))
    del v_ana, a_y_v_ana, v_sol, a_y_v_sol


# In[8]:


db["Prob2/Ell/Spec/V"] = rt
db.sync()


# In[9]:


n = 2048
eps_uzawa = 1.0e-10
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
nu_down, nu_up = 0, 1
iter_max = 100
l = 0
k = 4
rt = []


# In[10]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)


# In[11]:


for alpha in alphas:
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_uzawa_mg(n, alpha, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("alpha = {} finished".format(alpha))
    rt.append((alpha, ctr, elapsed, res, err))


# In[12]:


db["Prob2/VarAlpha"] = rt
db.sync()


# In[13]:


n = 2048
eps_uzawa = 1.0e-10
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
nus = [(2, 2)]
iter_max = 30
ls = [0, 1, 2, 3]
k = 4
rt = []


# In[14]:


for nu in nus:
    nu_down, nu_up = nu
    for alpha in alphas:
        for l in ls:
            u = numpy.zeros((n-1, n))
            v = numpy.zeros((n, n-1))
            p = numpy.zeros((n, n))
            u, v, p, ctr, elapsed = drivers.driver_uzawa_mg(n, alpha, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
            res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
            err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
            print("alpha = {}, nu = {}, l = {} finished".format(alpha, nu, l))
            rt.append((alpha, nu, l, ctr, elapsed, res, err))


# In[15]:


# db["Prob2/VarNuL(Alpha)"] = rt
db["Prob2/VarLAlpha"] = rt
db.sync()


# In[16]:


ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
eps_uzawa = 1.0e-10
alpha = 1.0
nu_down, nu_up = 0, 1
iter_max = 30
l = 0
k = 4
rt = []


# In[17]:


for n in ns:
    u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
    c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_uzawa_mg(n, alpha, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("n = {} finished".format(n))
    rt.append((n, l, ctr, elapsed, res, err))


# In[18]:


db["Prob2/VarN/Prob1"] = rt
db.sync()


# In[19]:


ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
eps_uzawa = 1.0e-10
alpha = 1.0
nu_down, nu_up = 0, 1
iter_max = 30
l = 0
k = 4
rt = []


# In[20]:


for n in ns:
    u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_2(n)
    c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)
    u = numpy.zeros((n-1, n))
    v = numpy.zeros((n, n-1))
    p = numpy.zeros((n, n))
    u, v, p, ctr, elapsed = drivers.driver_uzawa_mg(n, alpha, l, nu_down, nu_up, u, v, p, c_x, c_y, c_i, eps_uzawa, iter_max, k)
    res = models.sum_res(n, u, v, p, c_x, c_y, c_i)
    err = models.sum_err(n, u, v, p, u_ana, v_ana, p_ana)
    print("n = {} finished".format(n))
    rt.append((n, l, ctr, elapsed, res, err))


# In[21]:


db["Prob2/VarN/Prob2"] = rt
db.sync()


# In[22]:


db.close()

