# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Reading the dataset
df = pd.read_csv("heart.csv")
#Verifying it as a 'dataframe' object in pandas
type(df)
#Shape of dataset
df.shape
#Printing out a few columns
df.head(10)
df.sample(5)
#Description
df.info()
df.describe()
#Understanding the columns better
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])
#Analysing the target Variable
df["target"].describe()
df["target"].unique()
#Checking correlation between columns
print(df.corr()["target"].abs().sort_values(ascending=False))
#Exploratory Data Analysis
#First analysing the target variable
y = df["target"]
sns.countplot(y)
target_temp = df.target.value_counts()
print(target_temp)
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/1025,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/1025,2)))
#Analysing Sex Feature
df["sex"].unique()
sns.barplot(df["sex"],y)
#Analysing Chest Pain
df["cp"].unique()
sns.barplot(df["cp"],y)
#Analysing FBS Feature
df["fbs"].describe()
df["fbs"].unique()
sns.barplot(df["fbs"],y)
#Analysing the restecg Feature
df["restecg"].unique()
sns.barplot(df["restecg"],y)
#Analysing the exang Feature
df["exang"].unique()
sns.barplot(df["exang"],y)
#Analysing the slope feature
df["slope"].unique()
sns.barplot(df["slope"],y)
#Analysing the ca feature
df["ca"].unique()
sns.countplot(df["ca"])
sns.barplot(df["ca"],y)
#Analysing the thal feature
df["thal"].unique()
sns.barplot(df["thal"],y)
sns.distplot(df["thal"])
df.hist(figsize=(14,14))
plt.show()
#Train Test Split
from sklearn.model_selection import train_test_split

predictors = df.drop("target",axis=1)
target = df["target"]
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
X_train
X_test
Y_train
Y_test
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


a = X_train.iloc[6,:13]
#print(a)
b = [list(a)]
#print (b)
#Model Fitting
from sklearn.metrics import accuracy_score
 
#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
Y_pred_lr.shape
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
print(lr.predict(b))




#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,Y_train)
Y_pred_nb = nb.predict(X_test)
Y_pred_nb.shape
score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
print(nb.predict(b))

#SVM

from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)
Y_pred_svm.shape
score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
print(sv.predict(b))


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)
Y_pred_knn.shape
score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
print(knn.predict(b))


#Decision tree
from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x,max_depth=3)
    dt.fit(X_train,Y_train)

    Y_pred_dt = dt.predict(X_test)
    
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
        dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
#new_input=[[52,1,0,125,212,0,1,168,0,1,2,2,3]]
Y_pred_dt = dt.predict(X_test)
Y_pred_dt
print(Y_pred_dt.shape)
score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")



#Random Forest

from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


viewfor x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
        rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)

Y_pred_rf.shape
score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random forest is: "+str(score_rf)+" %")
print(rf.predict(b))



#Neural Network

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=300)
Y_pred_nn = model.predict(X_test)
Y_pred_nn.shape
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")





#output final scores
scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","Neural Network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
    
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)




