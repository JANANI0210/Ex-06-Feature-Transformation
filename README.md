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

df.isnull().sum()

![Screenshot (221)](https://user-images.githubusercontent.com/86832944/197984134-f1ba41ee-d32f-45d4-942f-64a7e0a66796.png)

![Screenshot (236)](https://user-images.githubusercontent.com/86832944/198814476-081c67e8-557d-4b10-9da9-80d81609e9c0.png)

df.info()

df.describe()

![Screenshot (237)](https://user-images.githubusercontent.com/86832944/198814501-fd07c455-bf8f-44ea-a677-9ec3ccce1ca3.png)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![Screenshot (238)](https://user-images.githubusercontent.com/86832944/198814576-5656b121-93e9-4344-8207-17f072a358aa.png)

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')

plt.show()

![Screenshot (248)](https://user-images.githubusercontent.com/86832944/198815012-5c8993ee-0588-49a3-bc1b-1421e53392f0.png)


sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')

plt.show()

![Screenshot (239)](https://user-images.githubusercontent.com/86832944/198815052-e4f80dcc-a32f-496a-9004-d5b5d641c2c1.png)


df4=df.copy()

df4['ModerateNegativeSkew_1'],parameters=stats.yeojohnson(df4.ModerateNegativeSkew)

![Screenshot (240)](https://user-images.githubusercontent.com/86832944/198815062-7b1f77e5-54bd-45e9-96c4-801bb83656ba.png)

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')

plt.show()


![Screenshot (241)](https://user-images.githubusercontent.com/86832944/198814764-7fe7c470-4819-4d35-89e1-d2e2b7992f83.png)

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)

plt.show()

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![Screenshot (242)](https://user-images.githubusercontent.com/86832944/198814779-a90f8605-8a47-4f78-a972-3dacef5335a4.png)

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![Screenshot (243)](https://user-images.githubusercontent.com/86832944/198814789-a25c7dd9-01ff-40f3-a0ab-62bb7796ca27.png)

df3 = df.copy()

df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![Screenshot (244)](https://user-images.githubusercontent.com/86832944/198814800-ee329c7c-4837-4c05-b70a-c49d07802c26.png)

df4 = df.copy()

df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)

sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')

plt.show()

![Screenshot (245)](https://user-images.githubusercontent.com/86832944/198814811-5df57b11-ba97-4775-ae88-6ebd11e1e48f.png)

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df4['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df4[['ModerateNegativeSkew']]))

sm.qqplot(df4['ModerateNegativeSkew_2'],fit=True,line='45')

plt.show()

![Screenshot (247)](https://user-images.githubusercontent.com/86832944/198814832-c59ef8f7-3d7d-48a7-ac97-9c2efc7d6aeb.png)

# RESULT

Thus the Feature Transformation for the given datasets had been executed successfully









