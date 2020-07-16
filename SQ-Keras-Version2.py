#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[4]:


print(data_info.loc['revol_util']['Description'])


# In[5]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[6]:


feat_info('mort_acc')


# In[8]:


df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[10]:


df.head()


# In[11]:


sns.countplot(x='loan_status',data=df)


# In[12]:


plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)


# In[14]:


df.corr()
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)


# In[15]:


feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(x='installment',y='loan_amnt',data=df,cmap='viridis')


# In[16]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# In[17]:


df.groupby('loan_status')['loan_amnt'].describe()


# In[18]:


sorted(df['grade'].unique())
sorted(df['sub_grade'].unique())


# In[19]:


sns.countplot(x='grade',data=df,hue='loan_status')


# In[26]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm',hue='loan_status')


# In[28]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


# In[29]:


df['loan_status'].unique()


# In[30]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[31]:


df[['loan_repaid','loan_status']]


# In[32]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# In[33]:


len(df)


# In[34]:


df.isnull().sum()


# In[35]:


100 * df.isnull().sum()/len(df)


# In[36]:


feat_info('emp_title')
print('\n')
feat_info('emp_length')


# In[37]:


df['emp_title'].nunique()
df['emp_title'].value_counts()


# In[38]:


df = df.drop('emp_title',axis=1)


# In[39]:


sorted(df['emp_length'].dropna().unique())


# In[40]:


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


# In[42]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# In[43]:


emp_co = df[df['loan_status']=='Charged Off'].groupby("emp_length").count()['loan_status']


# In[44]:


emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


# In[45]:


emp_len = emp_co/emp_fp


# In[46]:


emp_len.plot(kind='bar')


# In[47]:


df = df.drop('emp_length',axis=1)


# In[48]:


df = df.drop('title',axis=1)


# In[49]:


df['mort_acc'].value_counts()


# In[50]:


print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()


# In[51]:


print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']


# In[52]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[53]:


def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[54]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)


# In[55]:


df.isnull().sum()


# In[56]:


df = df.dropna()


# In[57]:


df.isnull().sum()


# In[58]:


df.select_dtypes(['object']).columns


# In[59]:


df['term'].value_counts()


# In[60]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))


# In[62]:


df['term'].value_counts()


# In[63]:


df = df.drop('grade',axis=1)


# In[64]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[65]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[66]:


df.select_dtypes(['object']).columns


# In[67]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[68]:


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')


# In[69]:


df['home_ownership'].value_counts()


# In[70]:


dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# In[72]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# In[73]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[74]:


df = df.drop('issue_d',axis=1)


# In[75]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# In[76]:


df.select_dtypes(['object']).columns


# In[77]:


from sklearn.model_selection import train_test_split


# In[78]:


df = df.drop('loan_status',axis=1)


# In[79]:


X = df.drop('loan_repaid',axis=1)


# In[80]:


y = df['loan_repaid'].values


# In[89]:


df = df.sample(frac=0.1,random_state=101)
print(len(df))


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[91]:


from sklearn.preprocessing import MinMaxScaler


# In[92]:


scaler = MinMaxScaler()


# In[93]:


X_train = scaler.fit_transform(X_train)


# In[94]:


X_test = scaler.transform(X_test)


# In[95]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# In[96]:


model = Sequential()

model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# In[97]:


model.fit(x=X_train,y=y_train,epochs=25,batch_size=256,validation_data=(X_test, y_test))


# In[99]:


losses = pd.DataFrame(model.history.history)


# In[100]:


losses.plot()


# In[101]:


from sklearn.metrics import classification_report,confusion_matrix


# In[103]:


predictions = model.predict_classes(X_test)


# In[106]:


print(classification_report(y_test,predictions))


# In[107]:


#imbalanced model, more "fully paid" than "charged off" loans


# In[110]:


df['loan_repaid'].value_counts()
31664/len(df)
# 80% of points already predicted as "loan repaid"... bottom threshold of 80%, so accuracy isn't great


# In[113]:


import random


# In[115]:


random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[117]:


new_customer = scaler.transform(new_customer.values.reshape(1,78))
#add extra bracket call in order to align with model structure & scaled data


# In[119]:


model.predict_classes(new_customer)


# In[121]:


df.iloc[random_ind]['loan_repaid']


# In[ ]:




