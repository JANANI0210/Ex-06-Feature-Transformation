# Ex-06-Feature-Transformation

## AIM
To read the given data and perform Feature Transformation process and save the data to a file. 

# Explanation
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file.

# CODE

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

df=pd.read_csv("/content/Data_to_Transform.csv")

print(df)

df.head()

![Screenshot (221)](https://user-images.githubusercontent.com/86832944/197984134-f1ba41ee-d32f-45d4-942f-64a7e0a66796.png)

