# %%
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import datetime

# %%
print('SPX Bid')
df_csv = pd.read_csv('renewable_bid_history_221018.csv')
df = df_csv.loc[:, ['id_1', 'id_5', 'id_7',  'id_11', 'price', 'amount']]
print(df.info())

# ランダムフォレスト
df_random_forest = pd.DataFrame(IterativeImputer(
    RandomForestRegressor(), max_iter=20, missing_values=np.nan).fit_transform(df))
df_random_forest.columns = df.columns
print('random forest')

line = 3
print('Line:', line)
print(df['id_1'].values[line])
print('id_1', df_random_forest['id_1'].values[line])
print('id_5', df_random_forest['id_5'].values[line])
print('id_7', df_random_forest['id_7'].values[line])
print('id_11', df_random_forest['id_11'].values[line])
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
d = now.strftime('%Y%m%d%H%M%S')
df_random_forest.to_csv('renewable_bid_history_'+d+'_rf.csv', index=False)
# %%
print("Complete!")
