# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:38:16 2020

@author: arpit kumar jain
"""

#importing libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('COVID19_DATA.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=0)

# Fitting Naive Bayes to the Training set
print("NAIVE BAYES")
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(xtrain, ytrain)




ypred=classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
print ('Accuracy Score :',accuracy_score(ytest, ypred)) 
print ('Report : ')
print (classification_report(ytest, ypred)) 


# Fitting Random Forest Classification to the Training set
print("random forest")
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(xtrain, ytrain)




ypred=classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
print ('Accuracy Score :',accuracy_score(ytest, ypred)) 
print ('Report : ')
print (classification_report(ytest, ypred)) 

# Fitting Kernel SVM to the Training set
print("kernal svm or svc")
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(xtrain, ytrain)



ypred=classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
print ('Accuracy Score :',accuracy_score(ytest, ypred)) 
print ('Report : ')
print (classification_report(ytest, ypred)) 


# Fitting SVM to the Training set
print("svm")
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(xtrain, ytrain)



ypred=classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
print ('Accuracy Score :',accuracy_score(ytest, ypred)) 
print ('Report : ')
print (classification_report(ytest, ypred)) 


# Fitting K-NN to the Training set
print("knn classfication")
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(xtrain, ytrain)


ypred=classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
print ('Accuracy Score :',accuracy_score(ytest, ypred)) 
print ('Report : ')
print (classification_report(ytest, ypred)) 

#LOGISTICREGRESSION
print("logistic regression")
from sklearn.linear_model import LogisticRegression
clas=LogisticRegression()
clas.fit(xtrain,ytrain)


ypred=clas.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
print ('Accuracy Score :',accuracy_score(ytest, ypred)) 
print ('Report : ')
print (classification_report(ytest, ypred)) 
