#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
sns.set()


# In[3]:


houses_df = pd.read_csv('houses.csv') 
houses_df.head()


# In[4]:


from pandas_profiling import ProfileReport
houses_price_profile=ProfileReport(houses_df,title='House Prices Profile')
houses_price_profile


# In[5]:


houses_df.shape


# In[6]:


houses_df.info()


# In[7]:


houses_df.isnull().sum()


# In[8]:


houses_df.describe().T


# In[9]:


houses_df.corr(method ='pearson') 


# In[10]:


houses_df['date'] = pd.to_datetime(houses_df['date'])
houses_df['date'].dtype


# In[12]:


houses_df['bedrooms'] = houses_df['bedrooms'].astype(int)
houses_df['bathrooms'] = houses_df['bathrooms'].astype(int)
houses_df['floors'] = houses_df['floors'].astype(int)


# In[13]:


houses_df.head()


# In[14]:


houses_df.dtypes


# In[15]:


for i in houses_df.columns:
    if ( houses_df[i].dtype ==  np.int64 ) | ( houses_df[i].dtype ==  np.int32):
        sns.lmplot(x = i, y ='price', data = houses_df)


# In[16]:


houses_df['date'].dt.year.unique(), houses_df['date'].dt.month.unique()


# In[17]:


houses_df['month']=  houses_df['date'].dt.month
houses_df.month.value_counts()


# In[18]:


houses_df.drop(['date'], axis= 1, inplace= True)


# In[19]:


houses_df.head(5)


# In[20]:


houses_df['country'].unique()


# In[21]:


houses_df['street'].value_counts()


# In[22]:


houses_df['city'].value_counts()


# In[23]:


houses_df['statezip'].value_counts()


# In[24]:


houses_df.drop(['country','street','statezip'], axis= 1, inplace=True)
houses_df.head(5)


# In[25]:


houses_df['bedrooms'].value_counts()


# In[26]:


houses_df=houses_df[(houses_df['bedrooms']>0) & (houses_df['bedrooms']<8)]
houses_df['bedrooms'].value_counts()


# In[27]:


houses_df.shape


# In[28]:


houses_df['bathrooms'].value_counts()


# In[29]:


sns.heatmap(houses_df[["price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]].corr(), annot=True);


# In[30]:


houses_df['floors'].value_counts()


# In[31]:


houses_df['waterfront'].value_counts()


# In[32]:


houses_df['view'].value_counts()


# In[33]:


houses_df['condition'].value_counts()


# In[34]:


sns.heatmap(houses_df[["price", "floors", "waterfront", "view", "condition"]].corr(), annot=True);


# In[35]:


houses_df['yr_built'].hist()


# In[36]:


houses_df['yr_renovated'].hist()


# In[37]:


sns.heatmap(houses_df[["price", "yr_built", "yr_renovated"]].corr(), annot=True);


# In[38]:


houses_df.head()


# In[39]:


houses_df['price'].describe().T


# In[40]:


houses_df.price.value_counts()


# In[41]:


houses_df.shape


# In[42]:


houses_df.price.plot()


# In[48]:


sns.boxplot(x='price', data= houses_df)


# In[43]:


def remove_outliers(df, x):
    # Set Limits
    q25, q75 = np.percentile(df[x], 25), np.percentile(df[x], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = 1 ,  (q75 + cut_off)
    df = df[(df[x] < upper) & (df[x] > lower)]
    print('Outliers of "{}" are removed\n'.format(x))
    return df


# In[44]:


houses_df= remove_outliers(houses_df, 'price')


# In[45]:


houses_df.shape


# In[49]:


houses_df.price.plot()


# In[50]:


sns.boxplot(x='price', data= houses_df);


# In[51]:


for i in houses_df.columns:
    if ( houses_df[i].dtype ==  np.int64 ) | ( houses_df[i].dtype ==  np.int32):
        sns.lmplot(x = i, y ='price', data = houses_df)


# In[52]:


df=houses_df
df= pd.get_dummies(df, drop_first=True)
df.head().T


# In[53]:


# Label
y= df.price.values
#Features
X = df.drop('price', axis =1).values


# In[54]:


#Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[55]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[56]:


#Mean and standard deviation of X_train
X_train.mean(), X_train.std()


# In[57]:


# Run Linear Regression Model and print accuracy
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print("Training acc >> ", lin_reg.score(X_train, y_train))
print("Testing acc >>  ", lin_reg.score(X_test, y_test))


# In[58]:


#Cross Val Score
lin_reg = LinearRegression()
cvscores_10 = cross_val_score(lin_reg, X, y, cv= 10)
print(np.mean(cvscores_10))


# In[59]:


rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(X_train, y_train)
print("Training acc >> ", rf_reg.score(X_train, y_train))
print("Testing acc >> ", rf_reg.score(X_test, y_test))


# In[ ]:




