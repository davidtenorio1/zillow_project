import pandas as pd 
import numpy as np 
import matplotlib as plt
import seaborn as sns
sns.set_style=("whitegrid")
import env
from env import user, password, host
import warnings
warnings.filterwarnings("ignore")

url = f'mysql+pymysql://{user}:{password}@{host}/zillow'

data = pd.read_sql('''select calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt from properties_2017
join predictions_2017 using (id)
join propertylandusetype using (propertylandusetypeid)
where transactiondate between "2017-05-01" and "2017-06-30"
and propertylandusetypeid not in ("31", "47", "246", "247", "248","264", "265", "266","267", "269", "270" )
''',url)

data