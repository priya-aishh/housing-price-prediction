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

training_data = pd.read_csv("dataset\train.csv")

test_data = pd.read_csv("dataset\test.csv")

#Correlation matrix
correlation_matrice = training_data.corr()
f, ax = plt.subplots( figsize=(15, 12))
sns.heatmap(correlation_matrice,vmin=0.2, vmax=0.8, square= True, cmap= 'BuPu')
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
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

x_train = data.drop('SalePrice',axis=1)
y_train = data['SalePrice']

scaler = preprocessing.StandardScaler() 
x_train_s = scaler.fit_transform(x_train)   

linear1 = LinearRegression()
linear1.fit(x_train_s, y_train)
pred = linear1.predict(x_train_s)
ax = sns.regplot(x=pred,y=y_train-pred,lowess=True,line_kws={"color":"black"})
ax.set_title('Figure 6 - Residual plot for original data.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.show();

print("Mean square error: ",mean_squared_error(y_train, pred))
print("Mean square error is very large,we need to use some log transformations")

y_train = np.log(y_train)
data['SalePrice']=np.log(data['SalePrice'])
linear1 = LinearRegression()
linear1.fit(x_train_s, y_train)
pred = linear1.predict(x_train_s)
ax = sns.regplot(x=pred,y=y_train-pred,lowess=True,line_kws={"color":"black"})
ax.set_title('Figure 7 - Residual plot for original data.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.show();

print("Mean square error: ",mean_squared_error(y_train, pred))
print("Root mean sq error:",sqrt(mean_squared_error(y_train, pred)))
