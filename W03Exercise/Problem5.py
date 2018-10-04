
# coding: utf-8

# In[ ]:


import time
import shelve
import numpy
import scipy.sparse
import scipy.sparse.linalg
import utils


# In[1]:


n_list = [5, 10, 15, 20, 30, 40, 50]


# In[ ]:


rt = [[[], [], []], [[], [], []], [[], [], []], [[], [], []]]


# In[ ]:


for n in n_list:
    
    x, y = numpy.indices((n, n))
    a = 1.0 / (1.0 + x + y)
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
    
    run(lambda: utils.solve_lu(a.copy(), b.copy()), 1)
    
    run(lambda: utils.solve_chol(a.copy(), b.copy()), 2)
    
    run(lambda: utils.solve_ldl(a.copy(), b.copy()), 3)
    
    print("n = {} finished".format(n))


# In[ ]:


with shelve.open("Result") as db:
    db["5size"] = n_list
    db["5result"] = rt

