#!/usr/bin/env python
# coding: utf-8

# Author :- Ashutosh Kumar
# Batch :- April
# Domain :- Data Science
# Aim :- To Build a model that Iris Flower Classification.

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv('IRIS.csv')


# In[3]:


df.head(25)


# In[4]:


df.shape


# In[5]:


df.tail(25)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.duplicated().sum()


# In[9]:


df.isna().sum()


# In[10]:


df.drop_duplicates(inplace= True)


# In[11]:


df.info()


# In[12]:


df


# In[13]:


valuecount = df['species'].value_counts().reset_index()
valuecount


# In[17]:


sns.lmplot(
    x="sepal_length",
    y="sepal_width",
    hue="species",
    palette="bright",
    data=df
)

plt.title("Sepal Length VS Sepal Width")
plt.show()


# In[18]:


sns.lmplot(
    x="petal_length",
    y="petal_width",
    hue="species",
    palette="bright",
    data=df
)

plt.title("Petal Length VS Petal Width")
plt.show()


# In[19]:


label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])


# In[20]:


X = df.drop(columns='species')
Y = df['species']


# In[21]:


X


# In[22]:


Y


# In[23]:


X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=43)


# In[24]:


X_train


# In[25]:


x_test


# In[26]:


Y_train


# In[27]:


y_test


# In[28]:


dfcorr = df.drop(columns = 'species', axis=1)


# In[29]:


dfcorr


# In[30]:


dfcorr.corr()


# In[31]:


plt.figure(figsize=(10, 8))  # Set the size of the heatmap
sns.heatmap(dfcorr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[32]:


logmodel = LogisticRegression()


# In[33]:


logmodel.fit(X_train,Y_train)


# In[34]:


logmodel.score(x_test,y_test) 


# In[35]:


logmodel.score(X_train, Y_train)


# In[36]:


logmodel.predict([[5.1,3.5,1.4,0.2]])


# In[ ]:




