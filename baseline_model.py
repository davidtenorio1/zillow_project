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
from math import sqrt
import split_scale
import evaluate
import env
from env import user, password, host
import warnings
warnings.filterwarnings("ignore")

url = f'mysql+pymysql://{user}:{password}@{host}/zillow'

data = pd.read_sql('''select id, calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt, taxvaluedollarcnt from properties_2017
join predictions_2017 using (id)
join propertylandusetype using (propertylandusetypeid)
where transactiondate between "2017-05-01" and "2017-06-30"
and propertylandusetypeid not in ("31", "47", "246", "247", "248","264", "265", "266","267", "269", "270" )
and calculatedfinishedsquarefeet * bathroomcnt * bedroomcnt != 0
and taxvaluedollarcnt != 0''',url)

data = data.set_index(data.id)

train, test = split_scale.split_my_data(data)

X_train = train.drop(columns=["id", "taxvaluedollarcnt"])
y_train = pd.DataFrame([train.taxvaluedollarcnt])
y_train = y_train.transpose()

X_train_scaled = split_scale.standard_scaler(X_train)


lr_mvp = LinearRegression()
lr_mvp = lr_mvp.fit(X_train_scaled, y_train)
yhat = lr_mvp.predict(X_train_scaled)
yhat = pd.DataFrame(yhat)


SSE_mvp = mean_squared_error(y_train, yhat)*len(y_train)
MSE_mvp = SSE_mvp/len(y_train)
RMSE_mvp = sqrt(MSE_mvp)
ESS = (yhat - y_train ** 2).sum()

baseline_data = train
baseline_data['mvp_yhat'] = lr_mvp.predict(X_train_scaled)
baseline_data['mvp_residuals'] = baseline_data.mvp_yhat - baseline_data.taxvaluedollarcnt
baseline_data['baseline_mean'] = baseline_data.taxvaluedollarcnt.mean()
baseline_data['baseline_residuals'] = baseline_data.baseline_mean - baseline_data.taxvaluedollarcnt
baseline_data











