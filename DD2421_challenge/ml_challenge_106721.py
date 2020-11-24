# -*- coding: utf-8 -*-
#Name : Shounak Chakraborty
#Email : shounakc@kth.se
#Personnummner : 19951113-5551
    
#--------------------------------------------------#

#1) IMPORT LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#Modeling:
from sklearn.ensemble import RandomForestClassifier


#Testing:
from sklearn.metrics import  accuracy_score

#--------------------------------------------------#

#2) DATA IMPORT AND PRE-PROCESSING

#import full data set
df = pd.read_csv('TrainOnMe.csv',sep=',') 
df_eval = pd.read_csv('EvaluateOnMe.csv',sep=',')

#list of relevant columns for model
col_list = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','y']
col_list_eval = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']

#dataframe created from list of relevant columns

df2 = df[col_list]
df2_eval = df_eval[col_list_eval]

df2.describe()
df2 = df2.dropna(how='any')
df2 = df2[df2.x1 != '?']
df2 = df2[df2.x5 != '?']

df2['x5'] = df2['x5'].replace({'TRUE':1, 'FALSE':0.5})

df2_eval = df2_eval.dropna(how='any')
df2_eval['x5'] = df2_eval['x5'].replace([True, False], [1, 0.5])

grade = {'Fx':0, 
            'F':1, 
            'E':2, 
            'D':3, 
            'C':4, 
            'B':5,
            'A':6}

# apply using map
X_6 = df2.x6.map(grade)
df2['x6'] = (X_6+1)/(max(X_6)+1)

X_6_eval = df2_eval.x6.map(grade)
df2_eval['x6'] = (X_6_eval+1)/(max(X_6_eval)+1)

#set X and Y:

X = df2.drop(['y'],axis=1).values #sets x and converts to an array

y = df2['y'].values #sets y and converts to an array

#split the data into train and test sets for numeric encoded dataset:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 10)

#--------------------------------------------------#

#3) MODELING AND TESTING:

#Numeric Encoded Model w/ SKLEARN:
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',
                                    random_state =43)
classifier.fit(X_train, y_train)

X_test_eval = df2_eval.values
y_pred_eval = classifier.predict(X_test_eval) # Predicting the Test set results

np.savetxt(r'106721.txt', y_pred_eval, fmt='%s')

y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred, normalize=False)) 

#--------------------------------------------------#


