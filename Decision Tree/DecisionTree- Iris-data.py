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


# In[6]:


sns.set_style('darkgrid')


# In[7]:


from sklearn.datasets import load_iris


# In[8]:


iris = load_iris()


# In[9]:


iris


# In[10]:


dataset = pd.DataFrame(iris.data , columns= iris.feature_names)


# In[11]:


dataset


# In[12]:


dataset['Target']= iris.target


# In[13]:


dataset


# In[16]:


X = dataset.drop(['Target'], axis=1)


# In[17]:





# In[18]:


y = dataset['Target']


# In[19]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[22]:


X_train.shape


# In[23]:


X_test.shape


# In[24]:


# decisioin tree


# In[25]:


from sklearn.tree import DecisionTreeClassifier


# In[26]:


classifier = DecisionTreeClassifier()


# In[27]:


classifier.fit(X_train, y_train)


# In[28]:


y_pred = classifier.predict(X_test)


# In[29]:


# check the predicrion


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


print(classification_report(y_test,y_pred))


# In[32]:


print(confusion_matrix(y_test, y_pred))


# In[33]:


from sklearn import tree


# In[34]:


plt.figure(figsize=(15,10))
tree.plot_tree(classifier)


# In[ ]:




