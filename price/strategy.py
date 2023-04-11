import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/result/price_result.csv')
df = pd.DataFrame(df)
# df.sort_values('date', inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
print(df['date'])
print(df['predict_price']) # means prediction made today for tomorrow open price
print(df['predict_price'].shift(1)) # means historical prediction for today's price

# build a naive long-short strategy, if the predict increase is larger than threshold%, long the bitcoin, otherwise short
threshold = np.arange(-0.05, 0.06, 0.01)
plt.figure(figsize=(14, 10))
for t in threshold:

    df['short_mask'] = ((df['predict_price'] - df['predict_price'].shift(-1)) / df['predict_price'].shift(-1) > t).astype('int')
    df['long_mask'] = ((df['predict_price'] - df['predict_price'].shift(-1)) / df['predict_price'].shift(-1) < -t).astype('int')
    # date in descend order, shift(-1) means shift upward, the decision make in today will gain tomorrow's return
    df['strategy_ret'] = df['return'].shift(-1) * (-df['short_mask'] + df['long_mask'])
    df['cum_ret'] = (df['return'] + 1).cumprod() - 1 # if holding the bitcoin for during the entire validation set
    df['cum_strategy_ret'] = (df['strategy_ret'] + 1).cumprod() - 1

    # df['short_mask'] = (
    #             (df['predict_price'] - df['predict_price'].shift(1)) / df['predict_price'].shift(1) < -t).astype('int')
    # df['long_mask'] = (
    #             (df['predict_price'] - df['predict_price'].shift(1)) / df['predict_price'].shift(1) > t).astype('int')
    # # date in ascending order, shift(-1) means shift upward, the decision make in today will gain tomorrow's return
    # df['strategy_ret'] = df['return'].shift(-1) * (-df['short_mask'] + df['long_mask'])
    # df['cum_ret'] = (df['return'] + 1).cumprod() - 1  # if holding the bitcoin for during the entire validation set
    # df['cum_strategy_ret'] = (df['strategy_ret'] + 1).cumprod() - 1
    #
    # df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    plt.plot(df['date'], df['cum_strategy_ret'], label=f'threshold={t:.2f}')

plt.plot(df['date'], df['cum_ret'], label='real')
plt.legend()
plt.ylabel('CUMULATIVE RETURN')
plt.xlabel('DATE')
plt.xticks(rotation=45)
plt.savefig(os.path.join('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/result', f'strategy_threshold_{threshold}.png'))
plt.show()

df = df[['date', 'open', 'predict_price', 'long_mask', 'short_mask', 'return', 'cum_ret', 'cum_strategy_ret']]

print(df.tail(20))
####################### single threshold
# threshold = 0.05
#
# df['short_mask'] = ((df['predict_price'] - df['predict_price'].shift(-1)) / df['predict_price'].shift(-1) > threshold).astype('int')
# df['long_mask'] = ((df['predict_price'] - df['predict_price'].shift(-1)) / df['predict_price'].shift(-1) < -threshold).astype('int')
# df['short_mask'] = df['short_mask'].shift(-1) # date in descend order, shift(-1) means shift upward, the decision make in today will gain tomorrow's return
# df['long_mask'] = df['long_mask'].shift(-1)
# df['strategy_ret'] = df['return'] * (-df['short_mask'] + df['long_mask'])
# df['cum_ret'] = (df['return'] + 1).cumprod() - 1 # if holding the bitcoin for during the entire validation set
# df['cum_strategy_ret'] = (df['strategy_ret'] + 1).cumprod() - 1
#
# df.dropna(inplace=True)
# df['date'] = pd.to_datetime(df['date'])
#
# plt.figure(figsize=(14, 10))
# plt.plot(df['date'], df['cum_strategy_ret'], label=f'threshold={threshold:.2f}')
# plt.plot(df['date'], df['cum_ret'], label='real')
# plt.legend()
# plt.ylabel('CUMULATIVE RETURN')
# plt.xlabel('DATE')
# plt.xticks(rotation=45)
# plt.savefig(os.path.join('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/result', f'strategy_threshold_{threshold}.png'))
# plt.show()

# df = df[['date', 'open', 'predict_price', 'long_mask', 'short_mask', 'return', 'cum_ret', 'cum_strategy_ret']]
# df.to_csv(os.path.join('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/result', f'strategy_threshold_{threshold}.csv'))
