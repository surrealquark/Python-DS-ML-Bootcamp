#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


loans = pd.read_csv('loan_data.csv')


# In[20]:


loans.describe()


# In[21]:


loans.head()


# In[22]:


loans.info()


# In[30]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[31]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[34]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans)


# In[41]:


plt.figure(figsize=(11,7))
sns.jointplot(x='fico',y='int.rate',data=loans)


# In[43]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')


# In[47]:


loans.info()


# In[60]:


cat_feats = ['purpose']


# In[61]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[62]:


final_data.info()


# In[63]:


from sklearn.model_selection import train_test_split


# In[65]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[66]:


from sklearn.tree import DecisionTreeClassifier


# In[67]:


dtree = DecisionTreeClassifier()


# In[69]:


dtree.fit(X_train,y_train)


# In[70]:


predictions = dtree.predict(X_test)


# In[71]:


from sklearn.metrics import classification_report,confusion_matrix


# In[73]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[74]:


from sklearn.ensemble import RandomForestClassifier


# In[75]:


rfc = RandomForestClassifier(n_estimators=200)


# In[77]:


rfc.fit(X_train,y_train)


# In[80]:


rfc_pred = rfc.predict(X_test)


# In[84]:


predictions = rfc.predict(X_test)


# In[86]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

