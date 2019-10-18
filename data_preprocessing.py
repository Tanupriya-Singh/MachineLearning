import numpy as np #To use mathematical functions
import matplotlib.pyplot as plt #pyplot is a particular function in matplotlib
import pandas as pd #Import datasets and manage datasets

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean",axis = 0) #axis means along which line to impute

imputer = imputer.fit(X[:, 1:3]) 
X[:, 1:3] = imputer.transform(X[:, 1:3]) 

#Encoding categorical data 
from sklearn.preprocessing import LabelEncoder
#New object of labelencoder class.
labelencoder_X = LabelEncoder()
#We will use this object on our data
labelencoder_X.fit_transform(X[:,0])

#Assign the encoded data to the original data
X[:,0 ] = labelencoder_X.fit_transform(X[:,0])

#Dummy variables for categorical variables. This is not required for the dependent variable.
from sklearn.preprocessing import OneHotEncoder
#Specify which column do you want
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray() #toarray converts the array returned into proper format

labelencoder_y = LabelEncoder()
labelencoder_y.fit_transform(y)

#We need to split the dataset into test set and training set
from sklearn.model_selection import train_test_split
#20% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
standardscaler_X = StandardScaler()
#For train set, we need to fit the train set to the scaler and then transform it
X_train = standardscaler_X.fit_transform(X_train)
#Here, we just need to transform the data
X_test = standardscaler_X.transform(X_test)

#X_train and X_test are scaled in the same way since the standardscaler_X is the same.
