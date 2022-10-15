#!/usr/bin/env python
# coding: utf-8

#                                            IBM ASSIGNMENT 2

# 1.IMPORTING THE REQUIRED PACKAGE

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 2. LOADING THE DATASET

# In[2]:


df = pd.read_csv("Churn_Modelling.csv")


# In[3]:


df


# 3. PERFORM VISUALIZATIONS

# 3.1 UNIVARIATE ANALYSIS

# In[4]:


sns.displot(df.Tenure)


# 3.2 BIVARIATE ANALYSIS

# In[5]:


df[df['CreditScore']<750].sample(750).plot.scatter(x='CreditScore',y='Age')


# In[32]:


sns.regplot(df['Age'],df['Tenure'],color='green')


# 3.3 MULTIVARIATE ANALYSIS

# In[7]:


categorical = df.drop(columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'])
rows = int(np.ceil(categorical.shape[1] / 2)) - 1
fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(10,6))
axes = axes.flatten()

for row in range(rows):
    cols = min(2, categorical.shape[1] - row*2)
    for col in range(cols):
        col_name = categorical.columns[2 * row + col]
        ax = axes[row*2 + col]       

        sns.countplot(data=categorical, x=col_name, hue="Exited", ax=ax);
        
plt.tight_layout()


# In[31]:


sns.pairplot(data=df[["RowNumber","Age","Tenure","Balance","NumOfProducts"]])


# 4. PERFORM DESCRIPTIVE STATISTICS ON THE DATASET

# In[8]:


df.info()


# In[10]:


df.describe()


# 5. HANDLE THE MISSING VALUES

# In[11]:


df.isna().sum()


# 6. FIND THE OUTLIERS AND REPLACE THE OUTLIERS

# In[12]:


outliers=df.quantile(q=(0.25,0.75))


# In[13]:


outliers


# In[14]:


# Finding inter-quartile range
q1 = df.CreditScore.quantile(0.25)
q3 = df.CreditScore.quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR


# In[15]:


# Median values
df.median(numeric_only=True)


# In[16]:


#Replacing the outliers
df['CreditScore'] = np.where(df['CreditScore'] < lower_limit, 7, df['CreditScore'])
sns.boxplot(x=df.CreditScore)


# 7. CHECKING FOR CATEGORICAL COLUMNS AND PERFORM ENCODING

# In[17]:


df.head()


# In[18]:


#Transforming Categorical columns into numerical values through labelencoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df.Gender = le.fit_transform(df.Gender)
df.Geography = le.fit_transform(df.Geography)

df.head()


# 8. SPLIT THE DATA INTO INDEPENDENT(X) AND INDEPENDENT(Y) VARIABLES

# In[19]:


x=df.drop(columns=["Surname"],axis=1)
x.head()


# In[20]:


y = df["Surname"]
y.head()


# 9. SCALE THE INDEPENDENT VARIABLES

# In[21]:


from sklearn.preprocessing import scale

X_Scaled = pd.DataFrame(scale(x), columns=x.columns)
X_Scaled.head()


# 10. SPLIT THE DATA INTO TRAINING AND TESTING

# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[25]:


print(x_train.shape)
print(x_test.shape)


# In[26]:


print(y_train.shape)
print(y_test.shape)


# In[27]:


x_train.head()


# In[28]:


x_test.head()


# In[29]:


y_train.head()


# In[30]:


y_test.head()


# In[ ]:




