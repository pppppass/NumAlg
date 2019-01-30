
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
import models
import drivers
import op


# In[2]:


n = 32


# In[3]:


u_ana_1, v_ana_1, p_ana_1, _, _, _ = models.get_models_1(n)
u_ana_2, v_ana_2, p_ana_2, _, _, _ = models.get_models_2(n)


# In[4]:


pyplot.figure(figsize=(8.0, 10.0))
pyplot.subplot(3, 2, 1)
pyplot.title("$u_1$")
pyplot.imshow(u_ana_1.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
pyplot.colorbar()
pyplot.subplot(3, 2, 2)
pyplot.title("$u_2$")
pyplot.imshow(u_ana_2.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
pyplot.colorbar()
pyplot.subplot(3, 2, 3)
pyplot.title("$v_1$")
pyplot.imshow(v_ana_1.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
pyplot.colorbar()
pyplot.subplot(3, 2, 4)
pyplot.title("$v_2$")
pyplot.imshow(v_ana_2.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
pyplot.colorbar()
pyplot.subplot(3, 2, 5)
pyplot.title("$p_1$")
pyplot.imshow(p_ana_1.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
pyplot.colorbar()
pyplot.subplot(3, 2, 6)
pyplot.title("$p_2$")
pyplot.imshow(p_ana_2.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
pyplot.colorbar()
pyplot.tight_layout()
pyplot.savefig("Figure1.pgf")
pyplot.show()


# In[5]:


def plot_var_res_fig(n, u, v, p, u_ana, v_ana, p_ana, c_x, c_y, c_i, filename):
    r_x = op.wrapper_add_b_x(n, p, op.wrapper_add_a_x(n, u, c_x.copy(), -1.0, 4), -1.0, 4)
    r_y = op.wrapper_add_b_y(n, p, op.wrapper_add_a_y(n, v, c_y.copy(), -1.0, 4), -1.0, 4)
    r_i = op.wrapper_add_b_y_t(n, v, op.wrapper_add_b_x_t(n, u, c_i.copy(), -1.0, 4), -1.0, 4)
    pyplot.figure(figsize=(8.0, 10.0))
    pyplot.subplot(3, 2, 1)
    pyplot.title("$ U - U^{\\star} $")
    pyplot.imshow((u - u_ana).transpose(), extent=(0.0, 1.0, 0.0, 1.0))
    pyplot.colorbar()
    pyplot.subplot(3, 2, 2)
    pyplot.title("$R_x$")
    pyplot.imshow(r_x.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
    pyplot.colorbar()
    pyplot.subplot(3, 2, 3)
    pyplot.title("$ V  - V^{\\star} $")
    pyplot.imshow((v - v_ana).transpose(), extent=(0.0, 1.0, 0.0, 1.0))
    pyplot.colorbar()
    pyplot.subplot(3, 2, 4)
    pyplot.title("$R_y$")
    pyplot.imshow(r_y.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
    pyplot.colorbar()
    pyplot.subplot(3, 2, 5)
    pyplot.title("$ P - P^{\\star} $")
    pyplot.imshow(((p - p_ana) - (p.mean() - p_ana.mean())).transpose(), extent=(0.0, 1.0, 0.0, 1.0))
    pyplot.colorbar()
    pyplot.subplot(3, 2, 6)
    pyplot.title("$R_{\\mathrm{i}}$")
    pyplot.imshow(r_i.transpose(), extent=(0.0, 1.0, 0.0, 1.0))
    pyplot.colorbar()
    pyplot.tight_layout()
    pyplot.savefig(filename)
    pyplot.show()


# In[6]:


u_ana, v_ana, p_ana, fs, cs, ds = models.get_models_1(n)
c_x, c_y, c_i = models.calc_rhs(n, fs, cs, ds)
u_sol, v_sol, p_sol, _, _ = drivers.driver_uzawa_mg(n, 1.0, 0, 0, 1, numpy.zeros((n-1, n)), numpy.zeros((n, n-1)), numpy.zeros((n, n)), c_x, c_y, c_i, 0.0, 2, 1, fftw=False)
_ = models.sum_res(n, u_sol, v_sol, p_sol, c_x, c_y, c_i, 4)
plot_var_res_fig(n, u_ana, v_ana, p_ana, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure2.pgf")


# In[7]:


t = numpy.linspace(0.0, 1.0, 2*n+1)
x_full, x_half = t[2:-1:2, None], t[1::2, None]
y_full, y_half = t[None, 2:-1:2], t[None, 1::2]


# In[8]:


u_init = (
      u_sol
    + numpy.sin(numpy.pi * 1.0 * x_full) * numpy.cos(numpy.pi * 1.0 * y_half)
    + numpy.sin(numpy.pi * (n*3//4) * x_full) * numpy.cos(numpy.pi * (n*5//7) * y_half)
)
v_init = (
      v_sol
    - numpy.cos(numpy.pi * 1.0 * x_half) * numpy.sin(numpy.pi * 2.0 * y_full)
    + numpy.sin(numpy.pi * (n*4//5) * x_half) * numpy.cos(numpy.pi * (n*2//3) * y_full)
)
p_init = (
      p_sol
    + numpy.cos(numpy.pi * 1.414 * x_half) * numpy.sin(numpy.pi * 0.314159 * y_half)
    - numpy.sin(numpy.pi * (n*0.866) * x_half) * numpy.cos(numpy.pi * (n*0.789) * y_half)
)
plot_var_res_fig(n, u_init, v_init, p_init, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure3.pgf")


# In[9]:


u, v, p, _, _ = drivers.driver_dgs_mg(n, 0, 0, 1, u_init.copy(), v_init.copy(), p_init.copy(), c_x, c_y, c_i, 0.0, 10, False, 1)
plot_var_res_fig(n, u, v, p, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure4.pgf")


# In[10]:


u, v, p, _, _ = drivers.driver_uzawa_mg(n, 0.8, 0, 0, 1, u_init.copy(), v_init.copy(), p_init.copy(), c_x, c_y, c_i, 0.0, 10, 1, fftw=False)
plot_var_res_fig(n, u, v, p, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure5.pgf")


# In[11]:


u, v, p, _, _ = drivers.driver_uzawa_inexact_cg_mg(n, 1.0, 1.0e-5, 1000, 0, 0, 1, u_init.copy(), v_init.copy(), p_init.copy(), c_x, c_y, c_i, 0.0, 10, 1)
plot_var_res_fig(n, u, v, p, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure6.pgf")


# In[12]:


u, v, p, _, _ = drivers.driver_uzawa_inexact_cg_mg(n, 1.0, 1.0e-5, 1000, 0, 0, 1, u_init.copy(), v_init.copy(), p_init.copy(), c_x, c_y, c_i, 0.0, 100, 1)
plot_var_res_fig(n, u, v, p, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure7.pgf")


# In[13]:


u, v, p, _, _ = drivers.driver_uzawa_inexact_pcg_mg_gs_mg(n, 1.0, 1.0e-5, 1000, 3, 4, 4, 0, 0, 1, u_init.copy(), v_init.copy(), p_init.copy(), c_x, c_y, c_i, 0.0, 100, 1)
plot_var_res_fig(n, u, v, p, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure8.pgf")


# In[14]:


u, v, p, _, _ = drivers.driver_uzawa_inexact_pcg_mg_gs_mg(n, 1.0, 1.0e-5, 1000, 2, 4, 4, 0, 0, 1, u_init.copy(), v_init.copy(), p_init.copy(), c_x, c_y, c_i, 0.0, 100, 1)
plot_var_res_fig(n, u, v, p, u_sol, v_sol, p_sol, c_x, c_y, c_i, "Figure9.pgf")

