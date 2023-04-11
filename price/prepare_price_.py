import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler



sen = pd.read_csv('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/result/sentiment_result.csv')
sen = pd.DataFrame(sen)
sen = sen.groupby('article_time')[['positive sentiment', 'negative sentiment']].agg('mean')\
    .reset_index().sort_values('article_time', ascending=False).reset_index(drop=True)
print(sen.head(20))
sen['article_time'] = pd.to_datetime(sen['article_time'], dayfirst=False)


def exponential_mv(prc_dt, df, label, a):
    numerator = 0
    denominator = 0
    # get the sentiment score according to the most recent 10 speeches
    mask = df['article_time'] <= prc_dt
    sentiment = df[mask][label][:10]
    delta = (pd.to_datetime(prc_dt) - pd.to_datetime(df[mask]['article_time'][:10])).dt.days
    for s, d in zip(sentiment, delta):
        numerator += (1 - a) ** d * s
        denominator += (1 - a) ** d
    try:
        wa = numerator / denominator
        return wa
    except:
        return None

## load sentiment
raw_path = '/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/dataset/btc_2.csv'
raw = pd.read_csv(raw_path)
raw = pd.DataFrame(raw)
raw = raw.sort_values('Date', ascending=True).reset_index(drop=True) #########
print(raw.head(20))
raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=False)
raw['positive sentiment'] = raw['Date'].apply(lambda prc_dt: exponential_mv(prc_dt, df=sen, label='positive sentiment', a=0.2))
raw['negative sentiment'] = raw['Date'].apply(lambda prc_dt: exponential_mv(prc_dt, df=sen, label='negative sentiment', a=0.2))
# print(prc.dropna().head(20))
raw.dropna(inplace=True)
raw.columns = raw.columns.str.lower()

#######
# mms = MinMaxScaler()
# xvar = ['open', 'return_lag_2', 'ma3', 'std3', 'log_vol', 'negative sentiment', 'positive sentiment', 'usdx']
# xvar = ['open']
# raw[xvar] = mms.fit_transform(raw[xvar])
# print(raw)
#
# scalerfile = '/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/mms.sav'
# pickle.dump(mms, open(scalerfile, 'wb'))

raw.to_csv(f'/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/dataset/raw.csv')




# ## make train and validation set

length = int(len(raw) * 0.8) # 0.8
raw_train = raw.iloc[:length] # latest 20% data are validation set
raw_val = raw.iloc[length:] # before the latest 20% are training set

raw_train.to_csv('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/dataset/train.csv')
raw_val.to_csv('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/dataset/val.csv')