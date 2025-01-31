# -*- coding: utf-8 -*-
"""Fish_Market_ML_Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UYeFMAvOxihM8oVyio4bol_e2BfkCdG0

## Importing the libraries
"""

import numpy as np
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Fish.csv')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

# print(X)
#
# print(y)

"""## Training the Polynomial Regression model on the whole dataset"""
#
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

"""##Predicting a new result with Polynomial Regression"""

# print(regressor.predict([[27.6, 30, 35.1, 14.0049, 4.84]]))

"""##Saving The Trained Model"""

import pickle
with open('fish_weight_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)