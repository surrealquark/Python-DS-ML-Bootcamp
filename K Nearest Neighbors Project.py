#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# In[29]:


import pandas as pd
import numpy as np


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


df = pd.read_csv('KNN_Project_Data',index_col=0)


# In[32]:


df.head()


# In[33]:


sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


# In[27]:


from sklearn.preprocessing import StandardScaler


# In[13]:


scaler = StandardScaler()


# In[20]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[21]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[22]:


scaled_features


# In[37]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[38]:


df.columns


# In[41]:


from sklearn.model_selection import train_test_split


# In[44]:


X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[46]:


from sklearn.neighbors import KNeighborsClassifier


# In[81]:


knn = KNeighborsClassifier(n_neighbors=17)


# In[82]:


knn.fit(X_train,y_train)


# In[83]:


pred = knn.predict(X_test)


# In[84]:


pred


# In[85]:


from sklearn.metrics import classification_report,confusion_matrix


# In[86]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[68]:


error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[80]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o')
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[71]:


error_rate


# In[94]:


knn = KNeighborsClassifier(n_neighbors=37)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




