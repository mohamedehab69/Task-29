#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns 
import datasist as ds
sns.set(rc={'figure.figsize':[10,10]},font_scale=1.2)


# In[2]:


df = pd.read_csv('train.csv')
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isna().sum()


# In[8]:


df.drop(['Cabin'],axis=1,inplace=True)


# In[9]:


df.isna().sum()


# In[13]:


df.info()


# In[10]:


from datasist.structdata import detect_outliers


# In[15]:


num_col= df.select_dtypes(include=['number']).columns
num_col


# In[17]:


for col in num_col:
    
    out_indices= detect_outliers(df,0,[col])
    print(f'column: {col}====>{len(out_indices)}outliers')
    df.loc[out_indices,col]=np.nan


# In[19]:


df.isna().sum()


# In[34]:


missing_col  = df.isna().sum().sort_values(ascending=False)[:1]
missing_col


# In[35]:


from sklearn.impute import KNNImputer


# In[37]:


imputer= KNNImputer()


# In[39]:


df.isna().sum()


# In[40]:


df.info()


# In[53]:


df= pd.get_dummies(df,  columns=['Name',  'Ticket'],drop_first=True)
df


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


x=df.drop('Survived',axis=1)
y=df['Survived']


# In[56]:


y.value_counts()


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[58]:


y_train.value_counts()


# In[59]:


from imblearn.under_sampling import RandomUnderSampler


# In[60]:


sampler = RandomUnderSampler()


# In[61]:


x_train,y_train = sampler.fit_resample(x_train,y_train)


# In[62]:


y_train.value_counts()


# In[63]:


from sklearn.preprocessing import StandardScaler


# In[64]:


scaler= StandardScaler()


# In[65]:


scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[66]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report


# In[67]:


models= {
    'LogisticRegression':LogisticRegression(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'SVC':SVC(),
    'GaussianNB':GaussianNB(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier(),
    'XGBClassifier':XGBClassifier()
}


# In[69]:


for name,model in models.items():
    print(f'Training Model {name}\n-----------')
    model.fit(x_train,y_train)
    y_pred= model.predict(x_test)
    print(f'Training Accuarcy:{accuracy_score(y_train,model.predict(x_train))}')
    print(f'Testing Accuarcy:{accuracy_score(y_test,y_pred)}')
    print(f'Testing Confusion Matrix: \n{confusion_matrix(y_test,y_pred)}')
    print(f'Testing Recall: {recall_score(y_test,y_pred)}')
    print(f'Testing precision: {precision_score(y_test,y_pred)}')
    print(f'Testing F-1: {f1_score(y_test,y_pred)}')


# In[ ]:




