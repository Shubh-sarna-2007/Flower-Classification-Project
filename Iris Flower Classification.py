#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score


# # Importing Dataset
# 

# In[ ]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length","sepal_width","petal_length","petal_width","class"]
iris_data = pd.read_csv(url,names=column_names)


# In[ ]:


iris_data.head(5)


# In[ ]:


type(iris_data)


# In[ ]:


iris_data.iloc[0:25]


# In[ ]:


iris_data.describe()


# In[ ]:


sns.pairplot(iris_data,hue="class") # colors based on different classes
plt.show()


# In[ ]:


X = iris_data.drop("class",axis=1)
X


# In[ ]:


y = iris_data["class"]
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42) # train the model


# In[ ]:


X_train


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3) # number of neighbours = 3
knn.fit(X_train,y_train)


# In[ ]:


y_pred = knn.predict(X_test) # check accuracy of model


# In[ ]:


print("Accuracy : ",accuracy_score(y_test, y_pred)) # accuracy is 100% success!!


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


y


# In[ ]:


X_test.head(2)


# In[ ]:


5.1	3.5	1.4	0.2	


# In[ ]:


new_data = pd.DataFrame({"sepal_length":[6.7],"sepal_width":6.5,"petal_length":3.4,"petal_width":2.9})


# In[ ]:


prediction = knn.predict(new_data)


# In[ ]:


prediction[0]


# In[ ]:


iris_data.iloc[100:]


# In[ ]:




