#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('whitegrid')



# In[2]:


import statsmodels.api as sm # use for OLS model


# In[3]:


# import Data which is available

data = pd.read_csv('sat.csv')


# In[4]:


data.head() #check the head


# In[5]:


data.describe() # check description


# In[6]:


y = data['GPA'] #dependent feature


# In[7]:


y.head() # check whether it load or not


# In[8]:


x1= data['SAT'] #independent feature


# In[9]:


# CHECK LINEAR RELATIONSHIP BETWEEN Independent & Dependent feature
plt.scatter(x1,y)
plt.show()


# In[10]:


## addd contasnt coef.
x=sm.add_constant(x1)


# In[11]:


x.head() # check wheather constant add or not


# In[12]:


result = sm.OLS(y,x).fit() # ordenary least square 1st dep. 2nd inde.


# In[13]:


result


# In[14]:


result.summary


# In[15]:


result.summary()


# In[16]:


plt.scatter(x1,y)

yhat = 0.275+0.0017*x1

plt.plot(x1,yhat, c='green')


# # multi linear regression

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[18]:


data = pd.read_csv('multisat.csv')


# In[19]:


data.head()


# In[20]:


# Y = b0 + b1x1 + b2x2
# GPA = B0 + B1*SAT + B2*RAND123


# In[21]:


z = data[['SAT', 'Rand 1,2,3']] # independent variable


# In[22]:


x = sm.add_constant(z)


# In[23]:


y = data['GPA'] # DEPENDENT VARIABLE


# In[24]:


result = sm.OLS(y,x).fit()


# In[25]:


result


# In[26]:


result.summary()


# In[27]:


result.summary2()


# In[28]:


x1 = data['Rand 1,2,3']

yhat = 0.2960+ 0.0017*x + (-0.0083)* x1


# In[29]:


plt.scatter(data['SAT'],y)


plt.show()


# In[ ]:




