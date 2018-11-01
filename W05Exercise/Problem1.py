
# coding: utf-8

# In[1]:


import numpy
import scipy.linalg
from utils import err


# In[2]:


n_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50]


# In[3]:


rt = [[], []]


# In[4]:


for n in n_list:
    a = scipy.linalg.hilbert(n)
    rt[0].append(err.est_cond_infty(n, a))
    rt[1].append(numpy.linalg.cond(a, numpy.infty))


# In[5]:


with open("Table1.tbl", "w") as f:
    for i in range(len(n_list)):
        f.write("{} & {:.5e} & {:.5e} \\\\\n".format(n_list[i], rt[0][i], rt[1][i]))
        f.write("\\hline\n")

