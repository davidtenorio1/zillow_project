import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols


def plot_residuals(x, y, dataframe):
    sns.set(style="whitegrid")
    return sns.residplot(x, y, color="b") 


def regression_errors(y, yhat):
    n = y.size
    residuals = yhat - y
    SSE = mean_squared_error(y, yhat)*n
    ESS = sum(residuals**2)
    TSS = ESS + SSE
    MSE = SSE/n
    RMSE = sqrt(MSE)
    return {
        'sse': SSE,
        'mse': SSE / n,
        'rmse': RMSE,
        'ess': ESS,
        'tss': TSS
    }


def baseline_mean_errors(yhat,y): 
    SSE_baseline = mean_squared_error(yhat, y)*len(y)
    MSE_baseline = SSE_baseline/len(y)
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









