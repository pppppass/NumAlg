
# coding: utf-8

# In[1]:


import shelve


# In[2]:


with shelve.open("Result") as db:
    n = db["size"]
    e_dir = db["direct"]
    e_col = db["colpivot"]
    e_full = db["fullpivot"]


# In[3]:


with open("Table1.tbl", "w") as f:
    for i in range(len(n)):
        f.write("{} & {:.5e} & {:.5e} & {:.5e} \\\\\n".format(n[i], e_dir[i], e_col[i], e_full[i]))
        f.write("\\hline\n")

