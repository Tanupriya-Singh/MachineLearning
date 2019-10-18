# Data Preprocessing

#Running on the run button will set this as the working directory

import numpy as np #To use mathematics functions
import matplotlib.pyplot as plt #pyplot is a particular function in matplotlib
import pandas as pd #Import datasets and manage datasets

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values # : means all the rows :-1 means all the columns barring the last one
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean",axis = 0)

#fit this object to X
imputer = imputer.fit(X[:, 1:3]) #The slicing operator does not include the higher values
X[:, 1:3] = imputer.transform(X[:, 1:3]) #Replaces the data with the mean.

