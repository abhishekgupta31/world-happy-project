#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('happiness_score_dataset.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.isnull().any()


# In[8]:


df.skew()


# In[9]:


df.shape


# In[10]:


df.columns


# In[11]:


sns.heatmap(df.isnull())


# In[ ]:


corr_mat =df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_mat,annot=True)



# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

df.hist(bins=50,figsize=(15,15))


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.groupby('Country')['Happiness Score'].max().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(12,6),color='yellow')
plt.title('Top 10 happiest countries')


# In[ ]:


df.groupby('Country')['Happiness Score'].max().sort_values(ascending=False).tail(10).plot(kind='bar', figsize=(12,6),color='red')
plt.title('Top 10 happiest countries')


# In[ ]:


df.groupby('Country')['Economy (GDP per Capita)'].max().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(12,6),color='green')
plt.title('Top 10 happiest countries')


# In[ ]:


df.groupby('Country')['Economy (GDP per Capita)'].max().sort_values(ascending=False).tail(10).plot(kind='bar', figsize=(12,6),color='green')
plt.title('Top 10 happiest countries')


# In[26]:


drop_rank = df.drop("Happiness Rank",axis = 1)


# In[30]:


dropped_happy = df.drop(["Country", "Happiness Rank",'Region'], axis=1)
dropped_happy.head()


# In[31]:


from sklearn.linear_model import LinearRegression
X = df.drop("Happiness Score", axis = 1)
lm = LinearRegression()
lm.fit(X, dropped_happy.Happiness Score)


# In[ ]:





# In[ ]:




