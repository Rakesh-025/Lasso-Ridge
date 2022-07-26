#Lasso ridege regression for life expectency data set
#Loading the packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# creating instance for one hot encoding
enc = OneHotEncoder()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# loading the data set into Python
life = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\lasso ridge\\Life_expectencey_LR.csv")
life.describe()

#Finding the missing values in the given data
life.isna().sum()

#Applying mean imputation for the missing values
life.fillna(life.mean(), inplace=True)
life.isna().sum()

#Applying linear regression model by using life expectancy as the output varaible
import statsmodels.formula.api as smf 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import pylab

final_ml = smf.ols('Life_expectancy ~ Year + Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling', data = life).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(life)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = life.Life_expectancy, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
life_train, life_test = train_test_split(life, test_size = 0.2)

# preparing the model on train data 
model_train = smf.ols('Life_expectancy ~ Year + Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling', data = life_train).fit()

# prediction on test data set 
test_pred = model_train.predict(life_test)

# test residual values 
test_resid = test_pred - life_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(life_train)

# train residual values 
train_resid  = train_pred - life_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

#Applying lasso ridge regression models
# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(life.iloc[:, 1:], life.Life_expectancy)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(life.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((lasso_pred - life.Life_expectancy)**2))



# Generating Ridge Regression model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(life.iloc[:, 1:], life.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(life.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((ridge_pred - life.Life_expectancy)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(life.iloc[:, 1:], life.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(life.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((enet_pred - life.Life_expectancy)**2))

