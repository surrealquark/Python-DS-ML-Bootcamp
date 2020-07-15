#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')


# In[15]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[16]:


feat_info('mort_acc')


# In[17]:


df = pd.read_csv('lending_club_loan_two.csv')


# In[18]:


df.head()


# In[20]:


sns.countplot(x='loan_status',data=df)


# In[22]:


sns.distplot(df['loan_amnt'],kde=False)


# In[26]:


df.corr()
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')


# In[32]:


feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(x='installment',y='loan_amnt',data=df,cmap='viridis')


# In[33]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# In[36]:


df.groupby('loan_status')['loan_amnt'].describe()


# In[38]:


df['grade'].unique()


# In[39]:


df['sub_grade'].unique()


# In[41]:


feat_info('sub_grade')


# In[42]:


sns.countplot(x='grade',data=df,hue='loan_status')


# In[54]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm',
             hue='loan_status')


# In[58]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,palette='coolwarm',hue='loan_status')


# In[62]:


df['loan_status'].unique()


# In[65]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[66]:


df[['loan_repaid','loan_status']]


# In[71]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# In[73]:


df.isnull().sum()


# In[84]:


100 * df.isnull().sum()/len(df)
df['emp_title'].nunique()
df['emp_title'].value_counts()


# In[85]:


df = df.drop('emp_title',axis=1)


# In[88]:


sorted(df['emp_length'].dropna().unique())


# In[99]:


emp_length_order = [ '< 1 year',
 '1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years',
 ]


# In[103]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# In[112]:


emp_co = df[df['loan_status']=='Charged Off'].groupby("emp_length").count()['loan_status']


# In[110]:


emp_fp = df[df['loan_status']=='Fully Paid'].groupby("emp_length").count()['loan_status']


# In[115]:


emp_len = emp_co/(emp_co + emp_fp)


# In[117]:


emp_len.plot(kind='bar')


# In[118]:


df = df.drop('emp_length',axis=1)


# In[125]:


df = df.drop('title',axis=1)


# In[128]:


df['mort_acc'].value_counts()


# In[129]:


df.corr()['mort_acc'].sort_values()


# In[134]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[136]:


def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[140]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)


# In[142]:


df.select_dtypes(['object']).columns


# In[165]:


feat_info('term')


# In[166]:


df['term'].value_counts()


# In[229]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))


# In[168]:


df['term'].value_counts()


# In[169]:


df = df.drop('grade',axis=1)


# In[170]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[171]:


df.select_dtypes(['object']).columns


# In[172]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[174]:


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')


# In[175]:


df['home_ownership'].value_counts()


# In[177]:


dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)


# In[181]:


df['address']


# In[183]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# In[184]:


df['zip_code'].value_counts()


# In[186]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = pd.concat([df.drop('zip_code',axis=1),dummies],axis=1)


# In[188]:


df = df.drop('address',axis=1)


# In[190]:


df = df.drop('issue_d',axis=1)


# In[203]:


df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))


# In[204]:


df['earliest_cr_line']


# In[205]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# In[206]:


df.select_dtypes(['object']).columns


# In[207]:


from sklearn.model_selection import train_test_split


# In[208]:


df = df.drop('loan_status',axis=1)


# In[209]:


X = df.drop('loan_repaid',axis=1)


# In[210]:


y = df['loan_repaid'].values


# In[212]:


df.sample(frac=0.1,random_state=101)
print(len(df))


# In[213]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[214]:


from sklearn.preprocessing import MinMaxScaler


# In[215]:


scaler = MinMaxScaler()


# In[217]:


X_train = scaler.fit_transform(X_train)


# In[218]:


X_test = scaler.transform(X_test)


# In[219]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# In[ ]:


model = Sequential()

model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )


# In[ ]:





# In[ ]:




