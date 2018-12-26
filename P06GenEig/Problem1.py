
# coding: utf-8

# In[53]:


import numpy
import scipy.sparse
from utils import models
from utils import geneig
from matplotlib import pyplot


# In[61]:


p = numpy.zeros(40)
p[-4], p[-1] = 1.0, 1.0
a = models.get_friend(p)


# In[62]:


h, f, it = geneig.driver_qr_impl(a.copy(), 1000)


# In[63]:


e = geneig.calc_eig_quasi_upper(h, f)


# In[68]:


pyplot.figure(figsize=(6.0, 6.0))
pyplot.scatter(e.real, e.imag, s=2.0)
pyplot.gca().set_aspect("equal")
pyplot.xlabel("$ \\Re \\lambda $")
pyplot.ylabel("$ \\Im \\lambda $")
pyplot.savefig("Figure1.pgf")
pyplot.show()
pyplot.close()


# In[65]:


v = geneig.calc_vec_all(a, e + 1.0e-15j, iters=1000)


# In[66]:


err_e, err_v = geneig.sum_eig_error((e, v), numpy.linalg.eig(a))


# In[67]:


with open("Table1.tbl", "w") as f:
    f.write("{} & {:.5e} & {:.5e} \\\\\n\\hline\n".format(it, err_e, err_v))

