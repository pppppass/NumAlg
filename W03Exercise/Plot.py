
# coding: utf-8

# In[1]:


import shelve
import numpy
import matplotlib
matplotlib.use("pgf")
from matplotlib import pyplot


# In[2]:


with shelve.open("Result") as db:
    n = db["4size"]
    rt = db["4result"]


# In[5]:


title = ["<LABEL1>", "<LABEL2>", "LU", "Cholesky", "LDL\\textsuperscript{T}", "Banded LU"]
pyplot.figure(figsize=(6.0, 4.5))
for i in range(6):
    pyplot.plot(n, rt[i][0], label=title[i])
    pyplot.scatter(n, rt[i][0], s=5.0)
pyplot.plot([numpy.power(10.0, 1.5), 1.0e2], [numpy.power(10.0, 0.5), numpy.power(10.0, 3.5)], linewidth=0.5, color="black", label="Slope $6$")
pyplot.legend()
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$N$")
pyplot.ylabel("<LABEL3>")
pyplot.savefig("Figure1.pgf")
pyplot.show()


# In[3]:


with open("Table6.tbl", "w") as f:
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
        f.write(("{} " + "& {:.5f} " * 4 + "\\\\\n").format(n[i], *(rt[j][0][i] for j in range(4))))
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

