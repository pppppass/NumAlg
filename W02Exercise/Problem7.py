
# coding: utf-8

# In[1]:


import numpy
import shelve


# In[2]:


def fact_lu(mat):
    n = mat.shape[0]
    a = mat
    for i in range(n-1):
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a


# In[3]:


def fact_lu_col_pivot(mat):
    n = mat.shape[0]
    a = mat
    p = numpy.arange(n)
    for i in range(n-1):
        j = numpy.argmax(numpy.abs(a[i:, i])) + i
        a[[i, j], :] = a[[j, i], :]
        p[[i, j]] = p[[j, i]]
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a, p


# In[4]:


def fact_lu_full_pivot(mat):
    n = mat.shape[0]
    a = mat
    p = numpy.arange(n)
    q = numpy.arange(n)
    for i in range(n-1):
        j, k = numpy.unravel_index(numpy.argmax(a[i:, i:]), (n-i, n-i))
        j, k = j+i, k+i
        a[[i, j], :] = a[[j, i], :]
        a[:, [i, k]] = a[:, [k, i]]
        p[[i, j]] = p[[j, i]]
        q[[i, k]] = q[[k, i]]
        a[i+1:, i] /= a[i, i]
        a[i+1:, i+1:] -= a[i+1:, i:i+1] * a[i:i+1, i+1:]
    return a, p, q


# In[5]:


def solve_upper(mat, vec):
    n = mat.shape[0]
    u, b = mat, vec
    for i in range(n-1, -1, -1):
        b[i] /= u[i, i]
        b[:i] -= b[i] * u[:i, i]
    return b


# In[6]:


def solve_lower_unit(mat, vec):
    n = mat.shape[0]
    l, b = mat, vec
    for i in range(n):
        b[i+1:] -= b[i] * l[i+1:, i]
    return b


# In[7]:


def solve_lu(mat, vec):
    lu, b = mat, vec
    solve_lower_unit(lu, b)
    solve_upper(lu, b)
    return b


# In[8]:


def solve_lu_perm(mat, vec, perm):
    lu, b, p = mat, vec, perm
    b = b[p]
    solve_lu(lu, b)
    return b


# In[9]:


def solve_lu_perm_double(mat, vec, perm_x, perm_y):
    lu, b, p, q = mat, vec, perm_x, perm_y
    b = b[p]
    solve_lu(lu, b)
    b = b[numpy.argsort(q)]
    return b


# In[10]:


n_list = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 84, 96, 128, 192, 256, 512, 1024, 2048]


# In[11]:


rt = [[], [], []]


# In[15]:


for n in n_list:
    
    x, y = numpy.indices((n, n))
    a = 6.0 * (x == y) + 1.0 * (x+1 == y) + 8.0 * (x == y+1)
    x = numpy.ones(n)
    b = a.dot(x)

    a_, p = fact_lu_col_pivot(a.copy())
    x_ = solve_lu_perm(a_, b.copy(), p)
    
    rt[0].append(numpy.linalg.norm(x - x_, numpy.infty))
    
    a_ = fact_lu(a.copy())
    x_ = solve_lu(a_, b.copy())
    
    rt[1].append(numpy.linalg.norm(x - x_, numpy.infty))
    
    a_, p, q = fact_lu_full_pivot(a.copy())
    x_ = solve_lu_perm_double(a_, b.copy(), p, q)
    
    rt[2].append(numpy.linalg.norm(x - x_, numpy.infty))
    
    print("n = {} finished".format(n))


# In[13]:


with shelve.open("Result") as db:
    db["size"] = n_list
    db["result"] = rt

