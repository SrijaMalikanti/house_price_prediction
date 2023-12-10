import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('Iris.csv')
data=data.drop(columns=['Id'])
data.head()
data.describe()
data.info()
data.isnull().sum()
data.hist()
#correlation between columns
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')

#LabelEncoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])
data.head(10)
from sklearn.model_selection import train_test_split
x=data.drop(columns=['Species'])
y=data['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)* 100)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)* 100)



from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)* 100)

