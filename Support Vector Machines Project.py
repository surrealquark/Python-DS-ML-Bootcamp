#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris = sns.load_dataset('iris')


# In[12]:


iris.head()


# In[13]:


iris.keys()


# In[14]:


sns.pairplot(data=iris,hue='species')


# In[42]:


setosa=iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],shade=True)


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[32]:


from sklearn.svm import SVC


# In[36]:


model = SVC()


# In[37]:


model.fit(X_train,y_train)


# In[38]:


predictions = model.predict(X_test)


# In[39]:


from sklearn.metrics import classification_report,confusion_matrix


# In[40]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[43]:


from sklearn.model_selection import GridSearchCV


# In[44]:


param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0,1,0.01,0.001,0.0001]}


# In[46]:


grid = GridSearchCV(SVC(),param_grid,verbose=3)


# In[47]:


grid.fit(X_train,y_train)


# In[48]:


grid.best_params_


# In[51]:


grid_predictions = grid.predict(X_test)


# In[54]:


print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))


# In[ ]:




