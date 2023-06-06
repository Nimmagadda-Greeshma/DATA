#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd                  # deals with data 
import numpy as np                   #deal with arrays 
import os                           # upload files 
import matplotlib.pyplot as plt    # display graphs  with requires specification 
import seaborn as sns             # graph module  in single line 


# In[3]:


df=pd.read_csv('Iris.csv')
df.head()


# In[4]:


df=df.drop(columns=['Id'])
df.tail()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df['Species'].value_counts()


# In[8]:


df.isnull().sum()


# In[9]:


df['SepalLengthCm'].hist()


# In[10]:


df['SepalWidthCm'].hist()


# In[11]:


df['PetalLengthCm'].hist()


# In[12]:


df['PetalWidthCm'].hist()


# # Two classes are merged together one is separate 

# In[13]:


colors=['red','green','blue']
species=['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[14]:



for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[15]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[16]:


# to reduce the number of variable 
# if two variables have high correlation value, one is dropped 
df.corr()


# In[17]:


corr=df.corr()
fig,ax =plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax, cmap='coolwarm')


# In[18]:


from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()


# In[19]:


df['Species']=le.fit_transform(df['Species'])
df.head()


# In[20]:


from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20)


# In[21]:


from sklearn.linear_model import LogisticRegression 
model=LogisticRegression()


# In[22]:


model.fit(x_train,y_train)


# In[23]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[25]:


model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:





# In[26]:





# In[ ]:





# In[ ]:




