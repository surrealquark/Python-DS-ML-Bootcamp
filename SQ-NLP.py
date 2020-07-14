#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


yelp = pd.read_csv('yelp.csv')


# In[9]:


yelp['text length'] = yelp['text'].apply(len)


# In[12]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length',bins=60)


# In[16]:


stars = yelp.groupby('stars').mean()
stars.corr()


# In[17]:


sns.heatmap(stars.corr(),annot=True)


# In[20]:


yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]
yelp_class.info()


# In[22]:


X = yelp_class['text']
y = yelp_class['stars']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[32]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)


# In[36]:


predictions = nb.predict(X_test)


# In[39]:


from sklearn.metrics import confusion_matrix,classification_report


# In[41]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[42]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[43]:


from sklearn.pipeline import Pipeline


# In[45]:


pipe = Pipeline([('bow',CountVectorizer()),
                 ('tfidf',TfidfTransformer()),
                 ('model',MultinomialNB())])


# In[54]:


X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[55]:


pipe.fit(X_train,y_train)


# In[56]:


predictionss = pipe.predict(X_test)


# In[58]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[ ]:




