
# coding: utf-8

# In[1]:


import numpy
import shelve
from utils import lu


# In[2]:


n_list = [2, 4, 8, 16, 32, 64, 84, 128, 256, 512]


# In[3]:


rt = [[], [], []]


# In[4]:


for n in n_list:
    
    i, j = numpy.indices((n, n))
    a = 6.0 * (i == j) + 1.0 * (i+1 == j) + 8.0 * (i == j+1)
    x = numpy.ones(n)
    b = a.dot(x)

    a_lu, p = lu.fact_lu_col(n, a.copy())
    x_sol = lu.solve_lu_col(n, a_lu, b.copy(), p)
    
    rt[0].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu = lu.fact_lu(n, a.copy())
    x_sol = lu.solve_lu(n, a_lu, b.copy())
    
    rt[1].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu, p, q = lu.fact_lu_full(n, a.copy())
    x_sol = lu.solve_lu_full(n, a_lu, b.copy(), p, q)
    
    rt[2].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    print("n = {} finished".format(n))


# In[5]:


with shelve.open("Result") as db:
    db["size"] = n_list
    db["result"] = rt

