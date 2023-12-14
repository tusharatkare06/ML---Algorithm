#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


sns.set_style('whitegrid')


# In[6]:


dataset =  pd.read_csv('Position_Salaries.csv')


# In[7]:


dataset


# In[8]:


X = dataset.iloc[:,1].values


# In[10]:


X


# In[11]:


y = dataset.iloc[:, 2].values


# In[12]:


y


# In[17]:


plt.scatter(X, y, color='green')
plt.show()


# In[18]:


from sklearn.tree import DecisionTreeRegressor


# In[20]:


regressor = DecisionTreeRegressor()


# In[21]:


regressor.fit(X.reshape(-1,1), y)


# In[22]:


plt.scatter(X, y, color = 'green')
plt.plot(X, regressor.predict(X.reshape(-1,1)), color='blue')


# In[23]:


# check the prediction

regressor.predict(np.array([6.5]).reshape(-1,1))


# In[ ]:





# In[ ]:




