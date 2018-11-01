
# coding: utf-8

# In[1]:


import numpy
import scipy.linalg
from utils import lu, ortho


# In[16]:


rt = [[[], [], [], [], [], []], [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []]]


# In[3]:


n_list = [2, 4, 8, 16, 32, 64, 84, 128, 256, 512]


# In[4]:


for n in n_list:
    
    i, j = numpy.indices((n, n))
    a = 6.0 * (i == j) + 1.0 * (i+1 == j) + 8.0 * (i == j+1)
    x = numpy.ones(n)
    b = a.dot(x)
    
    x_sol = numpy.linalg.solve(a, b)
    rt[0][0].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu = lu.fact_lu(n, a.copy())
    x_sol = lu.solve_lu(n, a_lu, b.copy())
    rt[0][1].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu, p = lu.fact_lu_col(n, a.copy())
    x_sol = lu.solve_lu_col(n, a_lu, b.copy(), p)
    rt[0][2].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu, p, q = lu.fact_lu_full(n, a.copy())
    x_sol = lu.solve_lu_full(n, a_lu, b.copy(), p, q)
    rt[0][3].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    q, r = numpy.linalg.qr(a)
    x_sol = numpy.linalg.solve(r, q.transpose().dot(b))
    rt[0][4].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_qr, beta = ortho.fact_qr_house(n, a.copy())
    x_sol = ortho.solve_qr(n, a_qr, beta, b.copy())
    rt[0][5].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    print("n = {} finished".format(n))


# In[5]:


with open("Table11.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], *(rt[0][j][i] for j in range(4))))
        f.write("\\hline\n")
with open("Table12.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], *(rt[0][j][i] for j in range(4, 6))))
        f.write("\\hline\n")


# In[6]:


n_list = [50, 100, 200, 300, 400, 500, 600, 700, 800]
numpy.random.seed(1)


# In[7]:


for n in n_list:
    
    i, j = numpy.indices((n, n))
    a = 10.0 * (i == j) + 1.0 * (i+1 == j) + 1.0 * (i == j+1)
    x = numpy.random.randn(n)
    b = a.dot(x)
    
    x_sol = numpy.linalg.solve(a, b)
    rt[1][0].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu = lu.fact_lu(n, a.copy())
    x_sol = lu.solve_lu(n, a_lu, b.copy())
    rt[1][1].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu, p = lu.fact_lu_col(n, a.copy())
    x_sol = lu.solve_lu_col(n, a_lu, b.copy(), p)
    rt[1][2].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu, p, q = lu.fact_lu_full(n, a.copy())
    x_sol = lu.solve_lu_full(n, a_lu, b.copy(), p, q)
    rt[1][3].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_llt = lu.fact_chol(n, a.copy())
    x_sol = lu.solve_chol(n, a_llt, b.copy())
    rt[1][4].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_ldlt = lu.fact_ldl(n, a.copy())
    x_sol = lu.solve_ldl(n, a_ldlt, b.copy())
    rt[1][5].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    q, r = numpy.linalg.qr(a)
    x_sol = numpy.linalg.solve(r, q.transpose().dot(b))
    rt[1][6].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_qr, beta = ortho.fact_qr_house(n, a.copy())
    x_sol = ortho.solve_qr(n, a_qr, beta, b.copy())
    rt[1][7].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    print("n = {} finished".format(n))


# In[8]:


with open("Table21.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], *(rt[1][j][i] for j in range(4))))
        f.write("\\hline\n")
with open("Table22.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], *(rt[1][j][i] for j in range(4, 8))))
        f.write("\\hline\n")


# In[17]:


n_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50]


# In[18]:


for n in n_list:
    
    a = scipy.linalg.hilbert(n)
    x = numpy.ones(n)
    b = a.dot(x)
    
    x_sol = numpy.linalg.solve(a, b)
    rt[2][0].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu = lu.fact_lu(n, a.copy())
    x_sol = lu.solve_lu(n, a_lu, b.copy())
    rt[2][1].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu, p = lu.fact_lu_col(n, a.copy())
    x_sol = lu.solve_lu_col(n, a_lu, b.copy(), p)
    rt[2][2].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_lu, p, q = lu.fact_lu_full(n, a.copy())
    x_sol = lu.solve_lu_full(n, a_lu, b.copy(), p, q)
    rt[2][3].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_llt = lu.fact_chol(n, a.copy())
    x_sol = lu.solve_chol(n, a_llt, b.copy())
    rt[2][4].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_ldlt = lu.fact_ldl(n, a.copy())
    x_sol = lu.solve_ldl(n, a_ldlt, b.copy())
    rt[2][5].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    q, r = numpy.linalg.qr(a)
    x_sol = numpy.linalg.solve(r, q.transpose().dot(b))
    rt[2][6].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    a_qr, beta = ortho.fact_qr_house(n, a.copy())
    x_sol = ortho.solve_qr(n, a_qr, beta, b.copy())
    rt[2][7].append(numpy.linalg.norm(x - x_sol, numpy.infty))
    
    print("n = {} finished".format(n))


# In[19]:


with open("Table31.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], *(rt[2][j][i] for j in range(4))))
        f.write("\\hline\n")
with open("Table32.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], *(rt[2][j][i] for j in range(4, 8))))
        f.write("\\hline\n")

