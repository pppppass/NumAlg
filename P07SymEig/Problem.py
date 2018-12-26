
# coding: utf-8

# In[1]:


import numpy
from utils import models, geneig, symeig
from matplotlib import pyplot


# In[2]:


def calc_log_line(start, end, intc, order):
    return [start, end], [intc, intc * (end / start)**order]


# In[3]:


def filter_array(array, lower):
    l = []
    for e in array:
        if numpy.abs(e) > lower:
            l.append(e)
        else:
            l.append(numpy.infty)
    return l


# In[4]:


ns = list(range(1, 101))
e_es, e_vs, its = [], [], []


# In[5]:


for n in ns:
    a = models.get_tri_const(n, 4.0, 1.0, 1.0)
    e, v, it = symeig.driver_thre_jacobi(a.copy(), 1.0, 1.0e5, 100)
    its.append(it)
    e_e, e_v = geneig.sum_eig_error((e, v), numpy.linalg.eig(a))
    e_es.append(e_e), e_vs.append(e_v)
    print("{} finished, {} iterations".format(n, it))


# In[8]:


pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(ns, its)
pyplot.scatter(ns, its, s=2.0)
pyplot.xlabel("$n$")
pyplot.ylabel("Iterations")
pyplot.savefig("Figure1.pgf")
pyplot.show()
pyplot.close()


# In[12]:


pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(ns, filter_array(e_es, 1.0e-16), label="Error of eigenvalues")
pyplot.scatter(ns, filter_array(e_es, 1.0e-16), s=2.0)
pyplot.plot(ns, filter_array(e_vs, 1.0e-16), label="Error of eigenvectors")
pyplot.scatter(ns, filter_array(e_vs, 1.0e-16), s=2.0)
pyplot.plot(*calc_log_line(3.0, 100.0, 0.5e-14, 1.0), linewidth=0.5, color="black", label="Slope $1$")
pyplot.plot(*calc_log_line(3.0, 100.0, 4.0e-15, 2.0), linewidth=0.5, color="black", linestyle="--", label="Slope $2$")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$n$")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.savefig("Figure2.pgf")
pyplot.show()
pyplot.close()

