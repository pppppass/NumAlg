
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import scipy.sparse
import scipy.sparse.linalg
import utils


# In[2]:


def get_func(size):
    n = size
    x, y = numpy.linspace(0.0, 1.0, n+2)[1:-1], numpy.linspace(0.0, 1.0, n+2)[1:-1]
    y, x = numpy.meshgrid(x, y)
    f = 2.0 * numpy.pi**2 * numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)
    f = f.reshape(-1)
    return f


# In[3]:


def get_sol(size):
    n = size
    x, y = numpy.linspace(0.0, 1.0, n+2)[1:-1], numpy.linspace(0.0, 1.0, n+2)[1:-1]
    y, x = numpy.meshgrid(x, y)
    s = numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)
    s = s.reshape(-1)
    return s


# In[4]:


def get_mat_sparse(size):
    n = size
    h = 1.0 / (n + 1)
    l = []
    for i in range(n):
        for j in range(n):
            l.append((4.0, i, j, i, j))
            l.append((-1.0, i, j, i, j-1))
            l.append((-1.0, i, j, i, j+1))
            l.append((-1.0, i, j, i-1, j))
            l.append((-1.0, i, j, i+1, j))
    a = utils.cvrt_list_to_csr(n, l)
    a = a / h**2
    return a


# In[5]:


def get_mat_dense(size):
    a = get_mat_sparse(size)
    a = a.todense().A
    return a


# In[6]:


n_list = [9, 19, 29]#, 39, 49, 59, 69, 79, 89, 99]


# In[10]:


rt = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]


# In[11]:


for n in n_list:
    
    def run(proc, index):
        start = time.time()
        s = proc()
        end = time.time()
        rt[index][0].append(end - start)
        rt[index][1].append(numpy.linalg.norm(s - s_ana, numpy.infty))
        rt[index][2].append(numpy.linalg.norm(s - s_eq, numpy.infty))
    
    f = get_func(n)
    s_ana = get_sol(n)
    
    a = get_mat_sparse(n)
    s_eq = scipy.sparse.linalg.spsolve(a, f)
    run(lambda: scipy.sparse.linalg.spsolve(a, f), 0)
    
    a = get_mat_dense(n)
    run(lambda: numpy.linalg.solve(a, f), 1)
    
    run(lambda: utils.solve_lu(a.copy(), f.copy()), 2)
    
    run(lambda: utils.solve_chol(a.copy(), f.copy()), 3)
    
    run(lambda: utils.solve_ldl(a.copy(), f.copy()), 4)
    
    run(lambda: utils.solve_lu_band(a.copy(), f.copy(), (n, n)), 5)
    
    print("n = {} finished".format(n))


# In[ ]:


with shelve.open("Result") as db:
    db["4size"] = n_list
    db["4result"] = rt

