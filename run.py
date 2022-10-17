# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# %%
print('SPX Bid')
df_csv = pd.read_csv('renewable_bid_history_221017.csv')
df = df_csv.loc[:, ['account_id', 'price', 'amount']]
df = df.fillna(0).astype('int')
print(df.info())
id = df['account_id'].values
print(id)
# price = df['price'].values
# print(price)
# amount = df['amount'].values

# ベイジアンリッジ
# df_bayesian_ridge = pd.DataFrame(IterativeImputer().fit_transform(df))
# df_bayesian_ridge.columns = df.columns
# print('bayesian ridge')
# print(df_bayesian_ridge['account_id'].values)

# ランダムフォレスト
df_random_forest = pd.DataFrame(IterativeImputer(
    RandomForestRegressor(), missing_values=0).fit_transform(df))
df_random_forest.columns = df.columns
print('random forest')
print(df_random_forest['account_id'].values)
df_random_forest.to_csv('renewable_bid_history_rf.csv', index=False)

print('#####################################################################')
print('UPX Ask')
df_csv = pd.read_csv('normal_ask_history_221017.csv')
df = df_csv.loc[:, ['account_id', 'price', 'amount']]
df = df.fillna(0).astype('int')
id = df['account_id'].values
print(id)

df_random_forest = pd.DataFrame(IterativeImputer(
    RandomForestRegressor(), missing_values=0).fit_transform(df))
df_random_forest.columns = df.columns
print('random forest')
print(df_random_forest['account_id'].values)
df_random_forest.to_csv('normal_ask_history_rf.csv', index=False)


# %%
print("Complete!")
