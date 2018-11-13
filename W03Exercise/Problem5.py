
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from utils import lu


# In[2]:


n_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50]


# In[3]:


rt = [[[], [], []], [[], [], []], [[], [], []], [[], [], []]]


# In[4]:


for n in n_list:
    
    a = scipy.linalg.hilbert(n)
    b = a.sum(axis=1)
    x = numpy.ones(n)
    
    def run(proc, index):
        start = time.time()
        x_ = proc()
        end = time.time()
        rt[index][0].append(end - start)
        rt[index][1].append(numpy.linalg.norm(x_ - x, numpy.infty))
        rt[index][2].append(numpy.linalg.norm(x_ - x, 2))
    
    run(lambda: numpy.linalg.solve(a, b), 0)
    
    run(lambda: lu.solve_lu(n, lu.fact_lu(n, a.copy()), b.copy()), 1)
    
    run(lambda: lu.solve_chol(n, lu.fact_chol(n, a.copy()), b.copy()), 2)
    
    run(lambda: lu.solve_ldl(n, lu.fact_ldl(n, a.copy()), b.copy()), 3)
    
    print("n = {} finished".format(n))


# In[5]:


with shelve.open("Result") as db:
    db["5size"] = n_list
    db["5result"] = rt

