
# coding: utf-8

# In[214]:


import numpy
import scipy.sparse
from utils import geneig
from matplotlib import pyplot


# In[215]:


a = numpy.array([
    [9.1, 3.0, 2.6, 4.0],
    [4.2, 5.3, 4.7, 1.6],
    [3.2, 1.7, 9.4, 0.0],
    [6.1, 4.9, 3.5, 6.2]
])


# In[216]:


xs = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 1.1, 1.2]


# In[217]:


class Result(object): pass
rt = Result()
rt.iters, rt.eigs, rt.errs = [], [], []


# In[218]:


for x in xs:
    a[2, 3] = x
    h, f, it = geneig.driver_qr_impl(a.copy(), 1000)
    rt.iters.append(it)
    e = geneig.calc_eig_quasi_upper(h, f)
    rt.eigs.append(e)
    v = geneig.calc_vec_all(a, e + 1.0e-15j, iters=1000)
    err = geneig.sum_eig_error((e, v), numpy.linalg.eig(a.astype(numpy.complex128)))
    rt.errs.append(err)


# In[219]:


pyplot.figure(figsize=(6.0, 4.0))
for i, x in enumerate(xs):
    pyplot.plot(rt.eigs[i].real, label="$ x = {:.1f} $".format(x))
pyplot.xlabel("Index")
pyplot.ylabel("$ \\Re \\lambda $")
pyplot.legend()
pyplot.savefig("Figure2.pgf")
pyplot.show()
pyplot.close()


# In[220]:


pyplot.figure(figsize=(6.0, 4.0))
for i, x in enumerate(xs):
    pyplot.plot(rt.eigs[i].imag, label="$ x = {:.1f} $".format(x))
pyplot.xlabel("Index")
pyplot.ylabel("$ \\Im \\lambda $")
pyplot.legend(loc="center left")
pyplot.savefig("Figure3.pgf")
pyplot.show()
pyplot.close()


# In[202]:


with open("Table2.tbl", "w") as f:
    for i, x in enumerate(xs):
        f.write("{:.1f} & {} & {:.5e} & {:.5e} \\\\\n\\hline\n".format(x, rt.iters[i], rt.errs[i][0], rt.errs[i][1]))

