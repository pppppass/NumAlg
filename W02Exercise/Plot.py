
# coding: utf-8

# In[1]:


import shelve


# In[2]:


with shelve.open("Result") as db:
    n = db["size"]
    rt = db["result"]


# In[3]:


with open("Table1.tbl", "w") as f:
    for i in range(len(n)):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} \\\\\n".format(n[i], rt[1][i], rt[0][i], rt[2][i]))
        f.write("\\hline\n")

