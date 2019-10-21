import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
sns.set_style('whitegrid')
import env
from env import user, password, host
import warnings
warnings.filterwarnings("ignore")

url = f'mysql+pymysql://{user}:{password}@{host}/zillow'

los_angeles_county_data = pd.read_sql('''select taxamount, taxvaluedollarcnt from properties_2017
join predictions_2017 using (id)
join propertylandusetype using (propertylandusetypeid)
where transactiondate between "2017-05-01" and "2017-06-30"
and propertylandusetypeid not in ("31", "47", "246", "247", "248","264", "265", "266","267", "269", "270" )
and taxdelinquencyflag is null
and fips like "6037"''',url)

orange_county_data = pd.read_sql('''select taxamount, taxvaluedollarcnt from properties_2017
join predictions_2017 using (id)
join propertylandusetype using (propertylandusetypeid)
where transactiondate between "2017-05-01" and "2017-06-30"
and propertylandusetypeid not in ("31", "47", "246", "247", "248","264", "265", "266","267", "269", "270" )
and taxdelinquencyflag is null
and fips like "6059"''',url)

ventura_county_data = pd.read_sql('''select taxamount, taxvaluedollarcnt from properties_2017
join predictions_2017 using (id)
join propertylandusetype using (propertylandusetypeid)
where transactiondate between "2017-05-01" and "2017-06-30"
and propertylandusetypeid not in ("31", "47", "246", "247", "248","264", "265", "266","267", "269", "270" )
and taxdelinquencyflag is null
and fips like "6111"''',url)

def drop_nulls_add_tax_rate_percent(data):
    data = data.dropna()
    data["tax_rate_percent"] = (data.taxamount / data.taxvaluedollarcnt) * 100
    return data, sns.distplot(data.tax_rate_percent)


drop_nulls_add_tax_rate_percent(los_angeles_county_data)

drop_nulls_add_tax_rate_percent(orange_county_data)

drop_nulls_add_tax_rate_percent(ventura_county_data)


