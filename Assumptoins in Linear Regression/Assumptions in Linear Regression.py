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


sns.set_style('darkgrid')


# In[6]:


## import the dataset in notebook

df = pd.read_csv('ad.csv', index_col=0)


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


# so sales is out dependent feature
y = df['Sales']


# In[11]:


x= df.drop(['Sales'], axis=1) # independent feature


# In[12]:


y.head()


# In[13]:


x.head()


# In[14]:


# so import statsmodel api for ad constant in this for OLS model


# In[15]:


import statsmodels.api as sm


# In[16]:


x_constant = sm.add_constant(x)


# In[17]:


x_constant.head()


# In[18]:


# lets train the linear regression model

lin_reg = sm.OLS(y ,x_constant)


# In[19]:


# fit the mdoel
lin_reg = lin_reg.fit()


# In[20]:


#lets check the summery


# In[21]:


lin_reg.summary()


# In[22]:


lin_reg.summary2()


# # 1st aasumptions is Linearity'
# 
# 

# In[23]:


#- scatter

sns.scatterplot(x=lin_reg.predict(x_constant), y=y)


# In[24]:


# regplot
sns.regplot(x=lin_reg.predict(x_constant), y=y)


# In[25]:


# rainbow test
sm.stats.diagnostic.linear_rainbow(lin_reg)


# In[26]:


x=lin_reg.predict(x_constant)


# In[27]:


x


# In[28]:


sns.scatterplot(data=df, x='TV', y='Sales')


# In[29]:


sns.scatterplot(data=df, x='Newspaper', y='Sales')


# In[30]:


sns.scatterplot(data=df, x='Radio', y='Sales')


# In[31]:


# aswe seen. relationship between in and outpur related to
# newspaper is no linear


# # 2sns assumption multicolinearity

# In[32]:


# correlation between inputs with each other


# In[ ]:





# In[33]:


df.drop(['Sales'], axis=1).corr()


# In[34]:


# heatmap
sns.heatmap(df.drop(['Sales'],axis=1).corr(), annot=True)


# In[35]:


# VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[36]:


vif=variance_inflation_factor


# In[37]:


x_values= x_constant.values


# In[38]:


x_values.shape


# In[39]:


VIF=[vif(x_values, i) for i in range(4)]


# In[40]:


VIF


# In[41]:


#VIF Range from1-5 moderate correlation
#VIF Range = 1 . then low correlation
#VIF Range = greater than 5 meangreater correlinearity


# In[42]:


df1= pd.DataFrame(VIF, index=x_constant.columns, columns= ['columns'])


# In[43]:


df1


# # 3. Multicoverient normality
# normalirt of residual

# In[44]:


residual = lin_reg.resid


# In[45]:


# KDE Plot
sns.kdeplot(residual)


# In[46]:


# Q-Q Plot

import statsmodels.api as sm


# In[47]:


sm.qqplot(residual, fit=True)


# In[48]:


# check skuwnes value

residual.skew()


# In[49]:


#skeewness is negative skew or left skew


# # 4. Homoscedesticity

# In[50]:


# we check homoscedesticity between residual and predicted value


# In[51]:


sns.scatterplot(x= lin_reg.predict(), y=residual)


# In[52]:


# using gold feldquand test


# In[53]:


import statsmodels.stats.api as sms


# In[54]:


sms.het_goldfeldquandt(lin_reg.resid, lin_reg.model.exog)


# In[55]:


# increasing, i.e. the variance in the second sample is larger than in the first, or decreasing or two-sided.


# # 5. Autocorrelation of the error
# 

# In[56]:


sns.lineplot(residual)


# In[57]:


plt.plot(residual)


# In[ ]:




