import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
sns.set_style('whitegrid')
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.feature_selection import SelectKBest, f_regression
from math import sqrt
import split_scale
import evaluate
import env
from env import user, password, host
import warnings
warnings.filterwarnings("ignore")


def select_kbest_freg(X_train, y_train, k):
    '''
    Takes data (X_train, y_train) and number of features to select (k) as input
    and returns a list of the top k features
    '''
    f_selector = SelectKBest(f_regression, k).fit(X_train, y_train).get_support()
    f_feature = X_train.loc[:,f_selector].columns.tolist()
    return f_feature