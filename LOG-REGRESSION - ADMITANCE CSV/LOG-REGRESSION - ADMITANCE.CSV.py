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


import statsmodels.api as sm


# In[6]:


sns.set_style('darkgrid')


# In[7]:


data = pd.read_csv('admitance.csv')


# In[8]:


data


# In[9]:


data2=data.copy()


# In[10]:


data.head()


# In[11]:


data['Admitted'].unique()


# In[12]:


data['Admitted']=data['Admitted'].map({'No':0, 'Yes':1} )


# In[13]:


data.head()


# In[14]:


# set dependent and independent variable
y= data['Admitted']


# In[15]:


x1= data['SAT']


# In[16]:


# check the linearity 


# In[21]:


sns.scatterplot( data= data,
    x='SAT',
    y='Admitted')


# # you can also plot using plt.scatter
# 
# plt.scatter(x1,y,color='red')
# 
# plt.xlabel('SAT')
# 
# plt.ylabel('Admited')
# 
# plt.show()

# In[22]:


# add constant coeeficient


# In[23]:


x =sm.add_constant(x1)


# In[24]:


x.head()


# In[26]:


# apply linear regression using OLS model statsmodel


# In[27]:


result = sm.OLS(y,x) # in OLS 1st Dependent then independent


# In[30]:


result = result.fit() # fit use to fit the data in model


# In[31]:


result.summary()


# In[32]:


result.params


# In[33]:


# yhat = b0+b1*x1


# In[34]:


yhat = (-3.251859)+(x1*0.002248)


# In[37]:


sns.scatterplot(x=x1, y=y)
sns.lineplot(x=x1, y=yhat)


# plt.scatter(x1,y,color='red')
# 
# yhat = x1*result_lin.params[1] + result_lin.params[0]   # 
# 
# x1*0.0022+(-3.251)
# 
# plt.plot(x1,yhat,lw=2.5,color='blue')
# 
# plt.xlabel('SAT')
# 
# plt.ylabel('Admited')
# 
# plt.show()

# In[38]:


# So linear regression not work here, we need to use logistic regression to train our odel


# In[40]:


reg_log = sm.Logit(y,x)


# In[41]:


reg_log= reg_log.fit()


# In[42]:


#so use the log formula to check 
# forumla =  exp. of (b0+b1x1)/1+exp(b0+b1x1)


# In[43]:


def formula(x,b0,b1):
    return np.array(np.exp(b0+x*b1)/(1+np.exp(b0+x*b1)))


# In[44]:


reg_log.params


# In[45]:


formula_result = formula(x1,(-69.912802),(0.042005))


# In[46]:


final_output = np.sort(formula_result)


# In[47]:


final_output.round(2)


# In[48]:


x_sort = np.sort(np.array(x1))


# In[49]:


sns.scatterplot(x=x1, y=y)
sns.lineplot(x=x_sort, y=final_output)


# In[50]:


plt.scatter(x1,y,color='red')
plt.plot(x_sort,final_output,lw=2.5,color='blue')
plt.xlabel('SAT')
plt.ylabel('Admited')
plt.show()


# In[ ]:




