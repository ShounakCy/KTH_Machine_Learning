#Name : Shounak Chakraborty
#Email : shounakc@kth.se
#Personnummner : 19951113-5551
#Temporarynumber : 19951113-T675
    
#--------------------------------------------------#

#1) IMPORT LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split

#Modeling:
from sklearn.ensemble import RandomForestClassifier

#Testing:
from sklearn.metrics import  accuracy_score

#--------------------------------------------------#

#2) DATA IMPORT AND PRE-PROCESSING

#import full data set
df = pd.read_csv('TrainOnMe.csv',sep=',') 

#list of relevant columns for model
col_list = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','y']

#dataframe created from list of relevant columns

df2 = df[col_list]
df2 = df2.dropna(how='any')
df2 = df2[df2.x1 != '?']
df2 = df2[df2.x5 != '?']

#Factorize dependent variable column:
X_5 = pd.factorize(df2['x5']) 
df2['x5'] = (X_5[0]+1)/(max(X_5[0])+1)
list_x5 = X_5[1] 

X_6 = pd.factorize(df2['x6'])
df2['x6'] = (X_6[0]+1)/(max(X_6[0])+1)
list_x6 = X_6[1] 

#set X and Y:

X = df2.drop(['y'],axis=1).values #sets x and converts to an array

y = df2['y'].values #sets y and converts to an array

#split the data into train and test sets for numeric encoded dataset:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 10)

#--------------------------------------------------#

#3) MODELING AND TESTING:

#Numeric Encoded Model w/ SKLEARN:
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',
                                    random_state =42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) # Predicting the Test set results
print(accuracy_score(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred, normalize=False)) 

#--------------------------------------------------#


