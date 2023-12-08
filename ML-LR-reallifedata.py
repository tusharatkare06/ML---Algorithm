#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')


# In[2]:


# load dataset 

raw_data = pd.read_csv("reallifedata.csv")


# In[3]:


raw_data.head()


# In[4]:


data = raw_data.drop(['Model'], axis=1)


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.describe(include='all')


# In[9]:


# dealing with null values
data.isnull()


# In[11]:


data.isnull().sum()


# In[12]:


data = data.dropna(axis=0)


# In[13]:


data.isnull().sum()


# In[14]:


# explore the data


# In[16]:


sns.distplot(data['Price'])
plt.show()


# In[18]:


sns.histplot(data['Price'], kde=True)
plt.show()


# In[19]:


sns.kdeplot(data['Price'])


# In[20]:


# as we observe that the distribution of price is not normal
# so we 


# In[21]:


plt.boxplot(data['Price'])


# In[22]:


# as we see lot muh outilesers there. so reduce this data 
# we need to quantile the price


# In[24]:


data = data[data['Price'] < data['Price'].quantile(0.99)]


# In[25]:


plt.boxplot(data['Price'])


# In[26]:


data = data[data['Price'] < data['Price'].quantile(0.99)]
plt.boxplot(data['Price'])


# In[27]:


data.describe()


# In[28]:


sns.kdeplot(data['Mileage'])


# In[31]:


data = data[data['Mileage'] < data['Mileage'].quantile(0.99)]

plt.boxplot(data['Mileage'])


# In[32]:


sns.kdeplot(data['Mileage'])


# In[33]:


data.describe()


# In[34]:


sns.kdeplot(data['EngineV'])


# In[35]:


plt.boxplot(data['EngineV'])


# In[36]:


# calculate outilesrs
q1, q3 = np.percentile(data['EngineV'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)


# In[37]:


lower_bound


# In[38]:


upper_bound


# In[41]:


data = data[data['EngineV']<4.8]


# In[42]:


plt.boxplot(data['EngineV'])


# In[43]:


sns.kdeplot(data['EngineV'])


# In[44]:


data.describe()


# In[45]:


sns.kdeplot(data['Year'])


# In[47]:


sns.boxplot(data['Year'])


# In[48]:


q1, q3 = np.percentile(data['Year'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)


# In[49]:


lower_bound


# In[50]:


data= data[data['Year']>1975]


# In[51]:


sns.boxplot(data['Year'])


# In[52]:


sns.kdeplot(data['Year'])


# In[53]:


data.describe()


# In[54]:


data


# In[55]:


final_data = data.reset_index(drop=True)


# In[56]:


final_data


# In[57]:


## check the linearity between data


# In[63]:


f,(ax1,ax2,ax3)= plt.subplots(1,3,sharey =True, figsize = (15,3))
ax1.scatter(final_data['Year'], final_data['Price'])
ax1.set_title('Year and Price')
ax2.scatter(final_data['EngineV'], final_data['Price'])
ax2.set_title('EngineV and Price')
ax3.scatter(final_data['Mileage'], final_data['Price'])
ax3.set_title('Mileage and Price')
plt.show()


# In[64]:


sns.kdeplot(final_data['Price'])


# In[65]:


# it seen like log distribution to convert it normal dist.


# In[66]:


log_price = np.log(final_data['Price'])


# In[67]:


sns.kdeplot(log_price)


# In[68]:


final_data['Log_Price']=log_price


# In[69]:


final_data


# In[70]:


f,(ax1,ax2,ax3)= plt.subplots(1,3,sharey =True, figsize = (15,3))
ax1.scatter(final_data['Year'], final_data['Log_Price'])
ax1.set_title('Year and Price')
ax2.scatter(final_data['EngineV'], final_data['Log_Price'])
ax2.set_title('EngineV and Price')
ax3.scatter(final_data['Mileage'], final_data['Log_Price'])
ax3.set_title('Mileage and Price')
plt.show()


# In[72]:


final_data = final_data.drop(['Price'],axis=1)


# In[73]:


final_data


# In[74]:


## check mulicolinearity


# In[75]:


final_data.columns


# In[76]:


final_data.info()


# In[77]:


final_data.dtypes


# In[82]:


variables = final_data[['Mileage', 'Year', 'EngineV']]


# In[89]:


variables


# In[84]:


# to check multicoliearity use VIF model import statsmodel


# In[90]:


variables.values


# In[85]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[86]:


vif=pd.DataFrame() # create vif datatframe


# In[88]:


variables.shape


# In[91]:


# vif['VIF'] mean creating column name VIF in vif dataframe

vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(3)]


# In[92]:


vif


# In[93]:


vif['Feature Name'] = variables.columns


# In[94]:


vif


# In[101]:


final_data_no_multico = final_data.drop(['Year'], axis=1)


# In[102]:


final_data_no_multico


# In[103]:


# dealing with categorical variablees


# In[104]:


final_data_no_multico.describe(include='all')


# In[105]:


# using one hot encoding convert categorical features to numerical features


# In[106]:


data_with_dummies = pd.get_dummies(final_data_no_multico, drop_first=True)


# In[107]:


data_with_dummies


# In[108]:


# dependent and independent features


# In[109]:


targets = data_with_dummies['Log_Price']
# dependent feature


# In[110]:


targets


# In[111]:


#independent feature
inputs = data_with_dummies.drop(['Log_Price'], axis=1)


# In[112]:


inputs


# In[114]:


# as we seen , every features have different unite and magnitude. so we need to standardscalar


# In[115]:


from sklearn.preprocessing import StandardScaler


# In[116]:


scaler = StandardScaler()


# In[117]:


scaler.fit(inputs)


# In[118]:


Final_input = scaler.transform(inputs)


# In[119]:


Final_input


# In[120]:


df = pd.DataFrame(Final_input)


# In[121]:


df


# In[125]:


# train test split the data , import libraries


# In[126]:


from sklearn.model_selection import train_test_split


# In[127]:


X_train, X_test, y_train, y_test = train_test_split(Final_input, targets, test_size=0.2, random_state=365)


# In[129]:


X_train.shape


# In[130]:


X_test.shape


# In[131]:


y_test.shape


# In[132]:


y_train.shape


# In[133]:


# Regression


# In[134]:


from sklearn.linear_model import LinearRegression


# In[135]:


reg=LinearRegression()


# In[136]:


reg.fit(X_train, y_train)


# In[137]:


yhat = reg.predict(X_train)


# In[144]:


plt.scatter(y_train, yhat)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# In[145]:


sns.kdeplot(y_train-yhat)


# In[146]:


reg.score(X_train, y_train) # Rsquare


# In[147]:


reg.intercept_


# In[148]:


reg.coef_


# In[149]:


reg_summary = pd.DataFrame(inputs.columns.values , columns=['Features'])


# In[150]:


reg_summary


# In[151]:


reg_summary['Slope']=reg.coef_


# In[152]:


reg_summary


# In[153]:


# Testing


# In[154]:


yhat_test = reg.predict(X_test)


# In[155]:


yhat_test


# In[156]:


y_test


# In[157]:


plt.scatter(y_test, yhat_test)


# In[ ]:




