import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm
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

url = f'mysql+pymysql://{user}:{password}@{host}/zillow'

data = pd.read_sql('''select id, calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt, taxvaluedollarcnt from properties_2017
join predictions_2017 using (id)
join propertylandusetype using (propertylandusetypeid)
where transactiondate between "2017-05-01" and "2017-06-30"
and propertylandusetypeid not in ("31", "47", "246", "247", "248","264", "265", "266","267", "269", "270" )
and calculatedfinishedsquarefeet * bathroomcnt * bedroomcnt != 0
and taxvaluedollarcnt != 0''',url)

data = data.set_index(data.id)

#sns.pairplot(data=data)

train, test = split_scale.split_my_data(data)

X_train = train.drop(columns=["id", "taxvaluedollarcnt"])
y_train = pd.DataFrame([train.taxvaluedollarcnt])
y_train = y_train.transpose()

X_test = test.drop(columns=["id", "taxvaluedollarcnt"])
y_test = pd.DataFrame([test.taxvaluedollarcnt])
y_test = y_test.transpose()

X_train_scaled = split_scale.standard_scaler(X_train)
#sns.heatmap(data.corr(), cmap='Blues', annot=True)

predictions=pd.DataFrame({'actual':y_train.taxvaluedollarcnt}).reset_index(drop=True)

# model 1 using square feet only
lm1=LinearRegression()
lm1.fit(X_train_scaled[['calculatedfinishedsquarefeet']],y_train)
lm1_predictions=lm1.predict(X_train_scaled[['calculatedfinishedsquarefeet']])
predictions['lm1']=lm1_predictions

# model 2 using square feet and bedroom count
lm2=LinearRegression()
lm2.fit(X_train_scaled[['calculatedfinishedsquarefeet', 'bedroomcnt']],y_train)
lm2_predictions=lm2.predict(X_train_scaled[['calculatedfinishedsquarefeet', 'bedroomcnt']])
predictions['lm2']=lm2_predictions

# model 3 using all three variables
lm3=LinearRegression()
lm3.fit(X_train_scaled[['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt']],y_train)
lm3_predictions=lm3.predict(X_train_scaled[['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt']])
predictions['lm3']=lm3_predictions

# model 4 using square feet and bathroom count
lm4=LinearRegression()
lm4.fit(X_train_scaled[['calculatedfinishedsquarefeet', 'bathroomcnt']],y_train)
lm4_predictions=lm4.predict(X_train_scaled[['calculatedfinishedsquarefeet', 'bathroomcnt']])
predictions['lm4']=lm4_predictions

# model 5 using bedroom count and bathroom count
lm5=LinearRegression()
lm5.fit(X_train_scaled[['bedroomcnt', 'bathroomcnt']],y_train)
lm5_predictions=lm5.predict(X_train_scaled[['bedroomcnt', 'bathroomcnt']])
predictions['lm5']=lm5_predictions

# model 6 using bedroom only
lm6=LinearRegression()
lm6.fit(X_train_scaled[['bedroomcnt']],y_train)
lm6_predictions=lm6.predict(X_train_scaled[['bedroomcnt']])
predictions['lm6']=lm6_predictions

# model 7 using bathroom only
lm7=LinearRegression()
lm7.fit(X_train_scaled[['bathroomcnt']],y_train)
lm7_predictions=lm7.predict(X_train_scaled[['bathroomcnt']])
predictions['lm7']=lm7_predictions

# baseline model
predictions['baseline'] = y_train.mean()[0]
predictions.head()

MSE_baseline = mean_squared_error(predictions.actual, predictions.baseline)
SSE_baseline = MSE_baseline*len(predictions.actual)
RMSE_baseline = sqrt(MSE_baseline)
r2_baseline = r2_score(predictions.actual, predictions.baseline)
print(MSE_baseline,SSE_baseline,RMSE_baseline,r2_baseline)

MSE_1 = mean_squared_error(predictions.actual, predictions.lm1)
SSE_1 = MSE_1*len(predictions.actual)
RMSE_1 = sqrt(MSE_1)
r2_1 = r2_score(predictions.actual, predictions.lm1)
print(MSE_1,SSE_1,RMSE_1,r2_1)

MSE_2 = mean_squared_error(predictions.actual, predictions.lm2)
SSE_2 = MSE_2*len(predictions.actual)
RMSE_2 = sqrt(MSE_2)
r2_2 = r2_score(predictions.actual, predictions.lm2)
print(MSE_2,SSE_2,RMSE_2,r2_2)

MSE_3 = mean_squared_error(predictions.actual, predictions.lm3)
SSE_3 = MSE_3*len(predictions.actual)
RMSE_3 = sqrt(MSE_3)
r2_3 = r2_score(predictions.actual, predictions.lm3)
print(MSE_3,SSE_3,RMSE_3,r2_3)

MSE_4 = mean_squared_error(predictions.actual, predictions.lm4)
SSE_4 = MSE_4*len(predictions.actual)
RMSE_4 = sqrt(MSE_4)
r2_4 = r2_score(predictions.actual, predictions.lm4)
print(MSE_4,SSE_4,RMSE_4,r2_4)

MSE_5 = mean_squared_error(predictions.actual, predictions.lm5)
SSE_5 = MSE_5*len(predictions.actual)
RMSE_5 = sqrt(MSE_5)
r2_5 = r2_score(predictions.actual, predictions.lm5)
print(MSE_5,SSE_5,RMSE_5,r2_5)

MSE_6 = mean_squared_error(predictions.actual, predictions.lm6)
SSE_6 = MSE_6*len(predictions.actual)
RMSE_6 = sqrt(MSE_6)
r2_6 = r2_score(predictions.actual, predictions.lm6)
print(MSE_6,SSE_6,RMSE_6,r2_6)

MSE_7 = mean_squared_error(predictions.actual, predictions.lm7)
SSE_7 = MSE_7*len(predictions.actual)
RMSE_7 = sqrt(MSE_7)
r2_7 = r2_score(predictions.actual, predictions.lm7)
print(MSE_7,SSE_7,RMSE_7,r2_7)



mvp_model=lm3.predict(X_test[['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt']])
mvp_model=mvp_model.ravel().reshape(3205)
y_test1=np.array(y_test).ravel().reshape(3205)
baseline_predictions = predictions['baseline'].ravel().reshape(3205)
best_model=pd.DataFrame({'model_predictions':mvp_model,'taxvaluedollarcnt':y_test1})

best_model.head()


sns.residplot(mvp_model, y_test)




test_results=pd.DataFrame({'actual':y_test.taxvaluedollarcnt}).reset_index(drop=True)
test_results['model_predictions'] = mvp_model
test_results['baseline_predictions'] = y_test.mean()[0]
sns.relplot(data=test_results)


