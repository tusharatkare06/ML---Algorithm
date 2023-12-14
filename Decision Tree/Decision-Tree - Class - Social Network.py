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


#load dataset

dataset = pd.read_csv('Social_Network_Ads.csv')


# In[7]:


dataset


# In[8]:


X = dataset.iloc[:,2:4].values


# In[9]:


y = dataset.iloc[:,4].values


# In[10]:


#train test split


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


# needstandard scalar 

from sklearn.preprocessing import StandardScaler


# In[16]:


sc = StandardScaler()


# In[17]:


X_train = sc.fit_transform(X_train)


# In[18]:


X_test = sc.transform(X_test)


# In[19]:


## decision treee


# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


classifier = DecisionTreeClassifier(criterion='entropy')


# In[22]:


classifier.fit(X_train, y_train)


# In[23]:


y_pred = classifier.predict(X_test)


# In[24]:


## check the report


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix


# In[27]:


print(classification_report(y_test, y_pred))


# In[28]:


print(confusion_matrix(y_test, y_pred))


# In[29]:


# check decision tree


# In[30]:


from sklearn import tree


# In[31]:


plt.figure(figsize=(15,10))
tree.plot_tree(classifier, filled= True)


# In[32]:


# Using GINI Impurity
classifier1 = DecisionTreeClassifier()


# In[33]:


classifier1.fit(X_train, y_train)


# In[34]:


y_pred1 =  classifier1.predict(X_test)


# In[35]:


print(classification_report(y_test, y_pred1))


# In[36]:


print(confusion_matrix(y_test, y_pred1))


# In[ ]:




