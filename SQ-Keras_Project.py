#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
    
feat_info('mort_acc')

df = pd.read_csv('lending_club_loan_two.csv')

df.head()

sns.countplot(x='loan_status',data=df)

sns.distplot(df['loan_amnt'],kde=False)

df.corr()
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')

feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(x='installment',y='loan_amnt',data=df,cmap='viridis')
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
df.groupby('loan_status')['loan_amnt'].describe()

df['grade'].unique()
df['sub_grade'].unique()
feat_info('sub_grade')

sns.countplot(x='grade',data=df,hue='loan_status')

plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm',
             hue='loan_status')

f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,palette='coolwarm',hue='loan_status')

df['loan_status'].unique()

df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

df[['loan_repaid','loan_status']]

df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')

df.isnull().sum()
100 * df.isnull().sum()/len(df)

df['emp_title'].nunique()
df['emp_title'].value_counts()
df = df.drop('emp_title',axis=1)


sorted(df['emp_length'].dropna().unique())

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

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

emp_co = df[df['loan_status']=='Charged Off'].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=='Fully Paid'].groupby("emp_length").count()['loan_status']
emp_len = emp_co/(emp_co + emp_fp)
emp_len.plot(kind='bar')

df = df.drop('emp_length',axis=1)
df = df.drop('title',axis=1)

df['mort_acc'].value_counts()

df.corr()['mort_acc'].sort_values()

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)
df.select_dtypes(['object']).columns

feat_info('term')

df['term'] = df['term'].apply(lambda term: int(term[:3]))
df['term'].value_counts()

df = df.drop('grade',axis=1)

subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)

df.select_dtypes(['object']).columns

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')
df['home_ownership'].value_counts()

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)

df['address']

df['zip_code'] = df['address'].apply(lambda address:address[-5:])

df['zip_code'].value_counts()

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = pd.concat([df.drop('zip_code',axis=1),dummies],axis=1)

df = df.drop('address',axis=1)

df = df.drop('issue_d',axis=1)

df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)
df.select_dtypes(['object']).columns

from sklearn.model_selection import train_test_split

df = df.drop('loan_status',axis=1)
X = df.drop('loan_repaid',axis=1)
y = df['loan_repaid'].values

df.sample(frac=0.1,random_state=101)
print(len(df))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm

model = Sequential()

model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )

#Anaconda crashed, will continue/debug tomorrow
