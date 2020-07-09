#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('College_Data',index_col=0)
df.head()


# In[24]:


df.describe()


# In[8]:


from sklearn.datasets import make_blobs


# In[15]:


sns.lmplot('Room.Board','Grad.Rate',data=df,hue='Private',palette='coolwarm')


# In[14]:


sns.lmplot('Room.Board','F.Undergrad',data=df,hue='Private',palette='coolwarm')


# In[16]:


from sklearn.cluster import KMeans


# In[20]:


kmeans = KMeans(n_clusters=2)


# In[25]:


kmeans.fit(df.drop('Private',axis=1))


# In[26]:


kmeans.cluster_centers_


# In[27]:


kmeans.labels_


# In[29]:


def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0
    


# In[33]:


df['Cluster'] = df['Private'].apply(converter)
df.head()


# In[34]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# In[36]:





# In[ ]:




