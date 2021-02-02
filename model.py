#!/usr/bin/env python
# coding: utf-8

# In[150]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,roc_curve
from sklearn.naive_bayes import GaussianNB
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[151]:


# Loading the preprocessed Dataset

df = pd.read_csv('cleaned_placement.csv')


# In[152]:


df.drop('gender',axis=1,inplace=True)
df.head()


# In[153]:


df.shape


# In[154]:


df['Avg per'] = (df['ssc_p']+df['hsc_p']+df['degree_p']+df['mba_p'])/5
df['Avg per']=df['Avg per'].round(2)


# In[155]:


plt.figure(figsize=(16,16))
sns.heatmap(df.corr(),annot=True)


# In[156]:


X = df.drop(['status'],axis=1) 
Y = df['status']
print(X.columns) # Features the Model will Train upon.
print(Y.head()) 
# spliting the data into 
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,shuffle=True,random_state=2)


# In[157]:


# We will use Logistic Regression Classification  model which will fit an S-curve i.e probability curve.


# In[158]:


X=X.reindex(columns= ['ssc_p', 'hsc_p', 'degree_p','mba_p','workex','Avg per'])
X.head()


# In[159]:


clf = GaussianNB()


# In[160]:


# Training  the Data 
clf.fit(x_train,y_train)


# In[161]:


# Predicting the value of Y i.e status on the test data.
y_pred = clf.predict(x_test)


# In[162]:


#pd.DataFrame({'Y Actual':y_test,'Y Pred':y_pred})


# In[163]:


# Confusion Matrix
conf_mat = pd.crosstab(y_test,y_pred)
conf_mat


# - Accuracy  0.8615384615384616
# - Recall  0.9523809523809523
# - Precision  0.851063829787234
# - F score   0.898876404494382

# In[164]:


accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
print("Accuracy ",accuracy)
print('Recall ',recall)
print('Precision ',precision)
print('F score  ',(2*recall*precision)/(recall+precision))


# In[165]:


fpr,tpr,thresholds = roc_curve(y_test,y_pred)
fpr,tpr,thresholds


# In[166]:


roc_score = roc_auc_score(y_test,y_pred)
roc_score


# In[167]:


# Pickle is used to save the model to be used for further use.
pickle.dump(clf,open('Model.pkl','wb'))


# In[ ]:





# In[168]:


X.head()


# In[169]:


model = pickle.load(open('Model.pkl','rb'))


# In[170]:


model.predict(x_test)


# In[171]:


model.score(x_test,y_test)


# In[172]:


df.head()


# In[173]:


df.shape


# In[174]:


# Predicting if a user gets the job 
# ssc_p=68,hsc_p=66,degree_p=90,mba_p=85,work_ex=0,etest_p=79


# In[ ]:




