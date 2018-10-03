
# coding: utf-8

# In[4]:


import shelve
import matplotlib
matplotlib.use("pgf")
from matplotlib import pyplot


# In[2]:


with shelve.open("Result") as db:
    n = db["4size"]
    rt = db["4result"]


# In[17]:


title = ["<LABEL1~~~>", "<LABEL2~~~>", "LU", "Cholesky", "LDL", "Banded LU"]
pyplot.figure(figsize=(6.0, 4.5))
for i in range(6):
    pyplot.plot(n, rt[i][0], label=title[i])
pyplot.legend()
pyplot.xlabel("$n$")
pyplot.ylabel("<LABEL3>")
pyplot.savefig("Figure1.pgf")
pyplot.show()


# In[3]:


with open("Table1.tbl", "w") as f:
    for i in range(len(n)):
        f.write(("{} " + "& {:.3f} " * 6 + "\\\\\n").format(n[i], *(rt[j][0][i] for j in range(6))))
        f.write("\\hline\n")


# In[4]:


with open("Table1.tbl", "w") as f:
    for i in range(len(n)):
        f.write(("{} " + "& {:.5e} " * 6 + "\\\\\n").format(n[i], *(rt[j][1][i] for j in range(6))))
        f.write("\\hline\n")


# In[5]:


with open("Table2.tbl", "w") as f:
    for i in range(len(n)):
        f.write(("{} " + "& {:.5e} " * 6 + "\\\\\n").format(n[i], *(rt[j][2][i] for j in range(6))))
        f.write("\\hline\n")


# In[6]:


with shelve.open("Result") as db:
    n = db["5size"]
    rt = db["5result"]


# In[7]:


with open("Table3.tbl", "w") as f:
    for i in range(len(n)):
        f.write(("{} " + "& {:.3f} " * 4 + "\\\\\n").format(n[i], *(rt[j][0][i] for j in range(4))))
        f.write("\\hline\n")


# In[8]:


with open("Table4.tbl", "w") as f:
    for i in range(len(n)):
        f.write(("{} " + "& {:.5e} " * 4 + "\\\\\n").format(n[i], *(rt[j][1][i] for j in range(4))))
        f.write("\\hline\n")


# In[9]:


with open("Table5.tbl", "w") as f:
    for i in range(len(n)):
        f.write(("{} " + "& {:.5e} " * 4 + "\\\\\n").format(n[i], *(rt[j][2][i] for j in range(4))))
        f.write("\\hline\n")

