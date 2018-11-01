
# coding: utf-8

# In[4]:


import numpy
import scipy.linalg
from utils import lu, err


# In[6]:


n_list = [5, 10, 15, 20, 25, 30, 40, 50]


# In[7]:


rt = [[], []]


# In[8]:


numpy.random.seed(1)


# In[10]:


for n in n_list:
    a = numpy.zeros((n, n))
    i, j = numpy.indices((n, n))
    a[i > j] = -1.0
    a[i == j] = 1.0
    a[j == n-1] = 1.0
    x = numpy.random.randn(n)
    b = a.dot(x)
    a_lu, p = lu.fact_lu_col(n, a.copy())
    x_sol = lu.solve_lu(n, a_lu, b.copy()[p])
    rt[0].append(err.est_error(n, a, b, x_sol))
    rt[1].append(numpy.linalg.norm(x - x_sol, numpy.infty))


# In[12]:


with open("Table2.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], rt[0][i], rt[1][i]))
        f.write("\\hline\n")

