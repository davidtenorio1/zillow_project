import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from pydataset import data
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

df = data("tips")
df['x'] = df[['total_bill']]
df['y'] = df[['tip']]
df['baseline'] = df['tip'].mean()
baseline = df['baseline']
x = pd.DataFrame(df['x'])
y = pd.DataFrame(df['y'])
y_mean = y.mean()*len(y)
ols_model = ols('y ~ x', data=df).fit()
df['yhat'] = ols_model.predict(x)
yhat = pd.DataFrame(df['yhat'])
ols_model.summary()


def plot_residuals(x, y, dataframe):
    sns.set(style="whitegrid")
    return sns.residplot(x, y, color="b") 


def regression_errors(y, yhat):
    SSE = mean_squared_error(df.y, df.yhat)*len(df)
    ESS = sum((df.yhat - df.y.mean())**2)
    TSS = ESS + SSE
    MSE = SSE/len(df)
    RMSE = sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(y): 
    SSE_baseline = mean_squared_error(df.y, df.baseline)*len(df)
    MSE_baseline = SSE_baseline/len(df)
    RMSE_baseline = sqrt(MSE_baseline)
    return SSE_baseline, MSE_baseline, RMSE_baseline


def better_than_baseline(y, yhat):
    SSE_baseline = baseline_mean_errors(y)[0]
    SSE_model = regression_errors(y, yhat)[0]
    return SSE_model < SSE_baseline


def model_significance(ols_model):
    r_squared = ols_model.rsquared
    r_pval = ols_model.f_pvalue
    return r_squared, r_pval









