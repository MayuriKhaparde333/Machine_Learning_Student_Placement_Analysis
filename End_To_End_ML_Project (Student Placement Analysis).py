#!/usr/bin/env python
# coding: utf-8

# # Machine Learning End To End Project (Student Placement Analysis)
# 

#  ##Steps that we have followed
# 
#  0. Preprocessing + EDA + Feature Selection
#  1. Extract input and output cols
#  2. Scale the values
#  3. Train test split
#  4. Train the model
#  5. Evaluate the model/model selection
#  6. Deploy the model

# In[45]:


#import libraries

import numpy as np
import pandas as pd


# In[2]:


# Importing the Dataset Student placement data

df = pd.read_csv('placement.csv')


# In[3]:


# show the head of the data

df.head()


# In[4]:


# information about the dataset

df.info()


# In[5]:


# Size of the Data

df.shape


# In[6]:


# preprocessing (Removing unwanted cloumns and rows)
   
f = df.iloc[:,1:]


# In[7]:


df.head()


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


# Plotting the Graph to understand the data (EDA)

plt.scatter(df['cgpa'],df['iq'],c=df['placement'])


# In[11]:


# Separating the indepent and dependent columns

X = df.iloc[:,0:2]
y = df.iloc[:,-1]


# In[12]:


X


# In[13]:


y.shape


# In[14]:


# Train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)


# In[15]:


X_train


# In[16]:


y_train


# In[17]:


X_test


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


# Scaling the data

scaler = StandardScaler()


# In[20]:


X_train = scaler.fit_transform(X_train)


# In[21]:


X_train


# In[22]:


X_test = scaler.transform(X_test)


# In[23]:


X_test


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


clf = LogisticRegression()


# In[26]:


# model training
clf.fit(X_train,y_train)


# In[27]:


y_pred = clf.predict(X_test)


# In[28]:


y_test


# In[29]:


from sklearn.metrics import accuracy_score


# In[30]:


accuracy_score(y_test,y_pred)


# In[40]:


get_ipython().system('pip install --upgrade mlxtend')


# In[41]:


from mlxtend.plotting import plot_decision_regions


# In[42]:


# Evaluate the Model

plot_decision_regions(X_train, y_train.values, clf=clf, legend=2)


# In[43]:


import pickle


# In[44]:


# Deploy the Model

pickle.dump(clf,open('model.pkl','wb'))


# In[ ]:




