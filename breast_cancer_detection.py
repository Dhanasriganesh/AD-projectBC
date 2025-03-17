#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the essental libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer


# In[5]:


cancer.keys()


# In[6]:


cancer.values()


# In[7]:


print(cancer['DESCR'])


# In[8]:


print(cancer['target'])


# In[9]:


print(cancer['target_names'])


# In[10]:


print(cancer['feature_names'])


# In[11]:


cancer['data'].shape


# In[12]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target']))


# In[13]:


df_cancer.head()


# In[14]:


df_cancer.tail()


# # *VISUALISING THE DATA*

# In[15]:


sns.pairplot(df_cancer , vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',
 'radius error', 'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness', 'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry', 'worst fractal dimension'])                                                                                                                                                                          


# In[16]:


sns.pairplot(df_cancer ,hue ='target', vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity']) 


# In[17]:


sns.countplot(df_cancer['target'])


# In[18]:


sns.scatterplot(x='mean area',y='mean smoothness',hue='target',data =df_cancer)


# In[19]:


plt.figure(figsize =(20,10))
sns.heatmap(df_cancer.corr(), annot =True)


# # *SPLITTING THE DATASET*

# In[20]:


x = df_cancer.drop(['target'],axis =1)


# In[21]:


x


# In[22]:


y= df_cancer['target']


# In[23]:


y


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


# In[26]:


x_train


# In[27]:


y_train


# In[28]:


x_test


# In[29]:


y_test


# # *TRAINING THE MODEL USING SVM*

# In[30]:


from sklearn.svm import SVC


# In[31]:


from sklearn.metrics import classification_report , confusion_matrix


# In[32]:


svc_model= SVC()


# In[33]:


svc_model.fit(x_train,y_train)


# # *EVALUATING THE MODEL*
# 

# In[34]:


y_predict =svc_model.predict(x_test)


# In[35]:


y_predict


# In[36]:


cm = confusion_matrix(y_test,y_predict)


# In[37]:


sns.heatmap(cm ,annot=True)


# # Model Improvisation
# 

# In[38]:


min_train =x_train.min()


# In[41]:


range_train =(x_train - min_train).max()


# In[42]:


x_train_scaled =(x_train-min_train)/range_train


# In[49]:


sns.scatterplot(x = x_train['mean area'], y= x_train['mean smoothness'],hue =y_train)


# In[50]:


sns.scatterplot(x = x_train_scaled['mean area'], y= x_train_scaled['mean smoothness'],hue =y_train)


# In[51]:


min_test =x_test.min()
range_test =(x_test - min_test).max()
x_test_scaled =(x_test-min_test)/range_test


# In[52]:


svc_model.fit(x_train_scaled,y_train)


# In[54]:


y_predict =svc_model.predict(x_test_scaled)


# In[55]:


cn = confusion_matrix(y_test,y_predict)


# In[56]:


sns.heatmap(cn,  annot = True)


# In[57]:


print(classification_report(y_test,y_predict))


# ### An accuracy of 96% has been achieved after appying the technique of Normalization for Improvisation

# In[63]:


param_grid ={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}


# In[65]:


from sklearn.model_selection import GridSearchCV


# In[67]:


grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[68]:


grid.fit(x_train_scaled,y_train)


# In[69]:


grid.best_params_


# In[70]:


grid_predictions=grid.predict(x_test_scaled)


# In[72]:


cn =confusion_matrix(y_test,grid_predictions)


# In[73]:


sns.heatmap(cn , annot =True)


# In[74]:


print(classification_report(y_test,grid_predictions))


# ### Accuracy of 97% has been achieved by further Improvisation by optimization of C and Gamma Parameters

# In[ ]:




