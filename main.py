# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:49:16 2019

@author: priya
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

training_data = pd.read_csv("dataset\\train.csv")

test_data = pd.read_csv("dataset\\test.csv")

#Correlation matrix
correlation_matrice = training_data.corr()
f, ax = plt.subplots( figsize=(15, 12))
sns.heatmap(correlation_matrice,vmin=0.1, vmax=0.6, square= True, cmap= 'OrRd')
plt.xlabel('The house features in the x axis',fontsize= 13)
plt.ylabel('The house features in the y axis',fontsize= 13)
plt.title('Fig 1 - The correlation matrix between all the featues ', fontsize= 16);
#We can see that TotRmsAbvGrd-GrLivArea and GarageYrBlt-YearBlt are highly correalated

#Checking linearity between all features
cols = ['SalePrice', 'HouseStyle','1stFlrSF','2ndFlrSF','OverallQual', 'TotalBsmtSF', 'YearBuilt', 'GrLivArea', 'GarageCars']
sns.pairplot(training_data[cols], size = 2.8)
plt.suptitle('Fig 2 - The scatter plot of the top features ',x=0.5, y=1.01, verticalalignment='top', fontsize= 18)
plt.tight_layout()
plt.show()

#Outlier detection in GrLivArea
# regplot of GrLivArea/SalePrice
ax = sns.regplot(x=training_data['GrLivArea'], y=training_data['SalePrice'])
plt.ylabel('SalePrice', fontsize= 10)
plt.xlabel('GrLivArea', fontsize= 10)
plt.title('Figure 3 - regplot of the GrLivArea with the SalePrice', fontsize= 12)
plt.show();
print("There are two outlier")

#removing outlier of GrLivArea
g_out = training_data.sort_values(by="GrLivArea", ascending = False).head(2)
print(g_out)
training_data.drop([523,1298], inplace = True)
training_data.reset_index(inplace=True)

# regplot of TotalBsmtSF/SalePrice
ax = sns.regplot(x=training_data['TotalBsmtSF'], y=training_data['SalePrice'])
plt.ylabel('SalePrice', fontsize= 13)
plt.xlabel('TotalBsmtSF', fontsize= 13)
plt.title('Figure 4 regplot of the TotalBsmtSF with the SalePrice', fontsize= 12);
plt.show()
print("There is no outlier in TotalBsmtSF")

# regplot of 1stFlrSF/SalePrice
ax = sns.regplot(x=training_data['1stFlrSF'], y=training_data['SalePrice'])
plt.ylabel('SalePrice', fontsize= 10)
plt.xlabel('1stFlrSF', fontsize= 10)
plt.title('Figure 5 - regplot of the 1stFlrSF with the SalePrice', fontsize= 12)
plt.show();
print("There is no outlier in 1stFlrSF")

#Treating missing values
print("Shape of training set: ", training_data.shape)
print("Missing values before remove NA: ")
print(training_data.columns[training_data.isnull().any()])

training_data.Alley.fillna(inplace=True,value='No')
training_data.BsmtQual.fillna(inplace=True,value='No')
training_data.BsmtCond.fillna(inplace=True,value='No')
training_data.BsmtExposure.fillna(inplace=True,value='No')
training_data.BsmtFinType1.fillna(inplace=True,value='No')
training_data.BsmtFinType2.fillna(inplace=True,value='No')
training_data.FireplaceQu.fillna(inplace=True,value='No') 
training_data.GarageType.fillna(inplace=True,value='No')
training_data.GarageFinish.fillna(inplace=True,value='No') 
training_data.GarageQual.fillna(inplace=True,value='No')    
training_data.GarageCond.fillna(inplace=True,value='No')
training_data.PoolQC.fillna(inplace=True,value='No')    
training_data.Fence.fillna(inplace=True,value='No')
training_data.MiscFeature.fillna(inplace=True,value='No')

#Numeric fields    
training_data.BsmtFinSF1.fillna(inplace=True,value=0)
training_data.BsmtFinSF2.fillna(inplace=True,value=0)
training_data.BsmtUnfSF.fillna(inplace=True,value=0)
training_data.TotalBsmtSF.fillna(value=0,inplace=True)
training_data.BsmtFullBath.fillna(inplace=True,value=0)
training_data.BsmtHalfBath.fillna(inplace=True,value=0)
training_data.GarageCars.fillna(value=0,inplace=True)
training_data.GarageArea.fillna(value=0,inplace=True)
training_data.LotFrontage.fillna(inplace=True,value=0)
training_data.GarageYrBlt.fillna(inplace=True,value=0)
training_data.MasVnrArea.fillna(inplace=True,value=0)

#Categorial fields
training_data.KitchenQual = training_data.KitchenQual.mode()[0]
training_data.Functional = training_data.Functional.mode()[0]
training_data.Utilities = training_data.Utilities.mode()[0]  
training_data.SaleType  = training_data.SaleType.mode()[0]
training_data.Exterior1st = training_data.Exterior1st.mode()[0]
training_data.Exterior2nd = training_data.Exterior2nd.mode()[0] 
training_data.Electrical = training_data['Electrical'].mode()[0]
training_data.MSZoning = training_data.MSZoning.mode()[0] 
training_data.MasVnrType=training_data['MasVnrType'].mode()[0]
print("After we imputed the missing values, the status of the data set is: ")
print(training_data.columns[training_data.isnull().any()])

#Mapping the ordinal fields which are strings to the corresponding meaningful codes.
#Transforming numeric categorical features to string.
#Applying one-hot encoding.
lotshape_map = {'Reg':'8','IR1':'6','IR2':'4','IR3':'2'}
training_data.LotShape = training_data.LotShape.map(lotshape_map)
training_data.LotShape = training_data.LotShape.astype('int64')

#Utilities: Type of utilities available       
utilities_map = {'AllPub':'8','NoSewr':'6','NoSeWa':'4','ELO':'2'}
training_data.Utilities = training_data.Utilities.map(utilities_map)
training_data.Utilities = training_data.Utilities.astype('int64')
    
#LandSlope: Slope of property
landslope_map = {'Gtl':'6','Mod':'4','Sev':'2'}
training_data.LandSlope = training_data.LandSlope.map(landslope_map)
training_data.LandSlope = training_data.LandSlope.astype('int64')

#ExterQual: Evaluates the quality of the material on the exterior 
quality_map = {'Ex':'10','Gd':'8','TA':'6','Fa':'4','Po':'2','No':'0'}
training_data.ExterQual = training_data.ExterQual.map(quality_map)
training_data.ExterQual = training_data.ExterQual.astype('int64')

#ExterCond: Evaluates the present condition of the material on the exterior
training_data.ExterCond = training_data.ExterCond.map(quality_map)
training_data.ExterCond = training_data.ExterCond.astype('int64')

#BsmtQual: Evaluates the height of the basement
training_data.BsmtQual = training_data.BsmtQual.map(quality_map)
training_data.BsmtQual = training_data.BsmtQual.astype('int64')

#BsmtCond: Evaluates the general condition of the basement
training_data.BsmtCond = training_data.BsmtCond.map(quality_map)
training_data.BsmtCond = training_data.BsmtCond.astype('int64')
#HeatingQC: Heating quality and condition
training_data.HeatingQC = training_data.HeatingQC.map(quality_map)
training_data.HeatingQC = training_data.HeatingQC.astype('int64')
        
#KitchenQual: Kitchen quality
training_data.KitchenQual = training_data.KitchenQual.map(quality_map)
training_data.KitchenQual = training_data.KitchenQual.astype('int64')

#FireplaceQu: Fireplace quality
training_data.FireplaceQu = training_data.FireplaceQu.map(quality_map)
training_data.FireplaceQu = training_data.FireplaceQu.astype('int64')

#GarageFinish: Interior finish of the garage
garage_map = {'Fin':'6', 'RFn':'4', 'Unf':'2', 'No':'0'}    
training_data.GarageFinish = training_data.GarageFinish.map(garage_map)
training_data.GarageFinish = training_data.GarageFinish.astype('int64')

#GarageQual: Garage quality
training_data.GarageQual = training_data.GarageQual.map(quality_map)
training_data.GarageQual = training_data.GarageQual.astype('int64')

#GarageCond: Garage condition
training_data.GarageCond = training_data.GarageCond.map(quality_map)
training_data.GarageCond = training_data.GarageCond.astype('int64')

#PoolQC: Pool quality
training_data.PoolQC = training_data.PoolQC.map(quality_map)
training_data.PoolQC = training_data.PoolQC.astype('int64')
#Converting numeric columns to nominal before applying one-hot encoding convertion
#After converting to String they will be treated as categorical
# MSSubClass as str
training_data['MSSubClass'] = training_data['MSSubClass'].astype("str")
# Year and Month to categorical
training_data['YrSold'] = training_data['YrSold'].astype("str")
training_data['MoSold'] = training_data['MoSold'].astype("str")
#Converting from str to int of ordinal fields
training_data.OverallCond = training_data.OverallCond.astype("int64")
training_data.OverallQual = training_data.OverallQual.astype("int64")
training_data['KitchenAbvGr'] = training_data['KitchenAbvGr'].astype("int64")

data = pd.get_dummies(training_data)
print("New  shape after one-hot encoding:" , np.shape(data))

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from math import sqrt


data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] + data['GarageArea']
x = data.drop('SalePrice',axis=1)
y = data['SalePrice']

scaler = preprocessing.StandardScaler()
x_train_s = scaler.fit_transform(x)   

linear1 = LinearRegression()
linear1.fit(x_train_s, y)
pred = linear1.predict(x_train_s)
ax = sns.regplot(x=pred,y=y-pred,lowess=True,line_kws={"color":"black"})
ax.set_title('Figure 5 - Residual plot for original data.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.show();

print("Mean square error: ",mean_squared_error(y, pred))
print("Root mean sq error:",sqrt(mean_squared_error(y, pred)))

print("from above result we can see that the mean sq error is very large we need to do some transformations on variables")
print("So we will do log transformation on traget variable to check the mse value if it has improved a bit")

data['SalePrice']=np.log(data['SalePrice'])
y = data['SalePrice']

linear2 = LinearRegression()
linear2.fit(x_train_s, y)
pred = linear2.predict(x_train_s)
ax = sns.regplot(x=pred,y=y-pred,lowess=True,line_kws={"color":"black"})
ax.set_title('Figure 6 - Residual plot for transformed data.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.show();

print("**************************************************************")

print("\nAfter cross validation")
kf = KFold(5, random_state=7, shuffle=True)

x = data.drop('SalePrice',axis=1).values
y = data['SalePrice'].values
for i, j in kf.split(x):
      x_train, x_test = x[i], x[j] 
      y_train, y_test = y[i], y[j]  

scaler = preprocessing.StandardScaler() 
x_train_s = scaler.fit_transform(x_train)  
x_test_s = scaler.fit_transform(x_test)

linear = LinearRegression()
linear.fit(x_train_s, y_train)
pred = linear.predict(x_train_s)

print("Mean square error: ",mean_squared_error(y_train, pred))
print("Root mean sq error:",sqrt(mean_squared_error(y_train, pred)))
errors = abs(pred - y_train)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors /y_train)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print("\n")

#Gradient Descent
print("After gradient descent")

def hypothesis(theta, X, n):
    h = np.ones((x_train_s.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,x_train_s.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(x_train_s.shape[0])
    return h

def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat

def sgd(train, alpha, n):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			sum_error += error**2
            cost[epoch] = sum_error
			coef[0] = coef[0] - alpha * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - alpha * error * row[i]
	return coef, cost

theta, cost = sgd(x_train_s, y_train,0.0001, 300000)
cost = list(cost)
n_iterations = [x for x in range(1,300001)]
plt.plot(n_iterations, cost)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.title("Line Curve Representation of Cost Minimization using SGD")

pred = hypothesis(theta,x_test_s,x_test_s.shape[1]-1)
print("Mean square error: ",mean_squared_error(y_test, pred))
print("Root mean sq error:",sqrt(mean_squared_error(y_test, pred)))
errors = abs(pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


from sklearn.ensemble import RandomForestRegressor

print("\n Random Forests ")

rf = RandomForestRegressor(n_estimators = 500, random_state = 42)

labels = np.array(y_train)
features = data.drop('SalePrice',axis=1)

# Train the model on training data
rf.fit(x_train, labels);
pred_rf = rf.predict(x_test) 

MSE_rf = mean_squared_error(y_test,pred_rf)
score_rf = np.sqrt(metrics.mean_squared_error(y_test,pred_rf))
RMSE_rf = score_rf
print("Mean square error: ",MSE_rf)
print("Root mean sq error:",RMSE_rf)

errors_rf = abs(pred_rf - y_test)
print('Mean Absolute Error:', round(np.mean(errors_rf), 2), 'degrees.')

mape_rf = 100 * (errors_rf/ y_test)
accuracy_rf = 100 - np.mean(mape_rf)
print('Accuracy:', round(accuracy_rf, 2), '%.')

print("Less and important features\n")
# Get numerical feature importances
feature_list = list(data.drop(['SalePrice'], axis=1).columns)
importances = list(rf.feature_importances_)
indices = np.argsort(importances)
# List of tuples with variable and importance
feature_importances = [(x_train, round(importance, 2)) for x_train, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
print("Importance of features in ascending order")
plt.barh(range(len(indices)),rf.feature_importances_[indices],color='b', align='center')
plt.show()
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
print("\n")

# New random forest with only the important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the most important features
important_indices = [feature_list.index('OverallQual'), feature_list.index('GrLivArea'), feature_list.index('TotalBsmtSF'), feature_list.index('GarageCars'), feature_list.index('GarageArea'), feature_list.index('YearBuilt'), feature_list.index('BsmtFinSF1'), feature_list.index('1stFlrSF'), feature_list.index('LotArea'),feature_list.index('OverallCond'),feature_list.index('YearRemodAdd'),feature_list.index('BsmtUnfSF'),feature_list.index('FireplaceQu'),feature_list.index('GarageYrBlt'),feature_list.index('GarageFinish'),feature_list.index('CentralAir_N'),feature_list.index('CentralAir_Y')]
train_important = x_train[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(x_test)
# Display the performance metrics
MSE_rf = mean_squared_error(y_test,predictions)
score_rf = np.sqrt(metrics.mean_squared_error(y_test,predections))
RMSE_rf = score_rf
print("Mean square error: ",MSE_rf)
print("Root mean sq error:",RMSE_rf)

errors_rf = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors_rf), 2), 'degrees.')
mape = np.mean(100 * (errors_rf / y_test))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

