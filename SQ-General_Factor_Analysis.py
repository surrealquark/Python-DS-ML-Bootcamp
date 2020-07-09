#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer = load_breast_cancer()


# In[8]:


cancer.keys()


# In[9]:


print(cancer['DESCR'])


# In[11]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[12]:


df.head()


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()
scaler.fit(df)


# In[18]:


scaled_data = scaler.transform(df)


# In[19]:


from sklearn.decomposition import PCA


# In[22]:


pca = PCA(n_components=2)


# In[23]:


pca.fit(scaled_data)


# In[24]:


x_pca = pca.transform(scaled_data)


# In[25]:


scaled_data.shape


# In[26]:


x_pca.shape


# In[32]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[34]:


pca.components_


# In[35]:


df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[36]:


df_comp


# In[37]:


plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')


# In[ ]:




