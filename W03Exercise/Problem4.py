
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
import scipy.sparse
import scipy.sparse.linalg
from utils import lu


# In[2]:


def cvrt_list_csr(size, list_):
    data, rows, cols = [], [], []
    for v, x1, x2, y1, y2 in list_:
        if 0 <= y1 < size and 0 <= y2 < size:
            data.append(v)
            rows.append(x1*size + x2)
            cols.append(y1*size + y2)
    mat = scipy.sparse.coo_matrix(
        (data, (rows, cols)), shape=(size**2, size**2))
    mat = mat.tocsr()
    return mat


# In[3]:


def get_func(size):
    n = size
    x, y = numpy.linspace(0.0, 1.0, n+1)[1:-1], numpy.linspace(0.0, 1.0, n+1)[1:-1]
    y, x = numpy.meshgrid(x, y)
    f = 2.0 * numpy.pi**2 * numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)
    f = f.reshape(-1)
    return f


# In[4]:


def get_sol(size):
    n = size
    x, y = numpy.linspace(0.0, 1.0, n+1)[1:-1], numpy.linspace(0.0, 1.0, n+1)[1:-1]
    y, x = numpy.meshgrid(x, y)
    s = numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)
    s = s.reshape(-1)
    return s


# In[5]:


def get_mat(size):
    n = size
    h = 1.0 / n
    l = []
    for i in range(n-1):
        for j in range(n-1):
            l.append((4.0, i, j, i, j))
            l.append((-1.0, i, j, i, j-1))
            l.append((-1.0, i, j, i, j+1))
            l.append((-1.0, i, j, i-1, j))
            l.append((-1.0, i, j, i+1, j))
    a = cvrt_list_csr(n-1, l)
    a = a / h**2
    return a


# In[6]:


n_list = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]


# In[7]:


rt = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]


# In[10]:


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
    
    a = get_mat(n)
    s_eq = scipy.sparse.linalg.spsolve(a, f)
    run(lambda: scipy.sparse.linalg.spsolve(a, f), 0)
    
    a = a.todense().A
    run(lambda: numpy.linalg.solve(a, f), 1)
    
    run(lambda: lu.solve_lu((n-1)**2, lu.fact_lu((n-1)**2, a.copy()), f.copy()), 2)
    
    run(lambda: lu.solve_chol((n-1)**2, lu.fact_chol((n-1)**2, a.copy()), f.copy()), 3)
    
    run(lambda: lu.solve_ldl((n-1)**2, lu.fact_ldl((n-1)**2, a.copy()), f.copy()), 4)
    
    run(lambda: lu.solve_lu_band((n-1)**2, lu.fact_lu_band((n-1)**2, a.copy(), (n, n)), f.copy(), (n, n)), 5)
    
    print("n = {} finished".format(n))


# In[9]:


with shelve.open("Result") as db:
    db["4size"] = n_list
    db["4result"] = rt

