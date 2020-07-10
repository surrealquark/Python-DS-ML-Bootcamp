#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[8]:


columns_names = ['user_id','item_id','rating','timestamp']


# In[9]:


df = pd.read_csv('u.data',sep='\t',names=columns_names)


# In[10]:


df.head()


# In[11]:


movie_titles = pd.read_csv('Movie_Id_Titles')


# In[12]:


movie_titles.head()


# In[13]:


df = pd.merge(df,movie_titles,on='item_id')


# In[14]:


df.head()


# In[15]:


sns.set_style('white')


# In[16]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[17]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[18]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[19]:


ratings.head()


# In[20]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[21]:


ratings.head()


# In[22]:


ratings['num of ratings'].hist(bins=70)


# In[23]:


ratings['rating'].hist(bins=70)


# In[24]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# In[25]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')


# In[26]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# In[27]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']


# In[28]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[29]:


similar_to_starwars.head()


# In[30]:


similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# In[31]:


similar_to_liarliar


# In[32]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# In[37]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[41]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar.head()


# In[42]:


corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar.head()


# In[43]:


corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




