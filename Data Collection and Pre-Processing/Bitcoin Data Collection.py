#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:57:21 2023

@author: lancelotpan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# get Bitcoin trading data from Yahoo Finance
btc = yf.download('BTC-USD', start='2014-09-17', end='2023-02-28')

btc['return'] = np.log(btc['Adj Close']/btc['Adj Close'].shift(1))
btc['cumulative return'] =  (btc['return'] + 1).cumprod()

# plt.plot(btc.index, btc['cumulative return'])


# lag 2,4,8

# Calculating the price moving average and other variables
window = 3

btc['MA' + str(window)] = btc['Adj Close'].rolling(window).mean().shift(1).values
btc['std'+str(window)] = btc['Adj Close'].rolling(window).std().shift(1).values

btc['std5'] = btc['Adj Close'].rolling(5).std().shift(1).values
btc['return_lag_2'] = btc['return'].shift(2).values
btc['return_lag_2_dummy'] = np.sign(btc['return_lag_2'])
btc['return_lag_4'] = btc['return'].shift(4).values
btc['return_lag_4_dummy'] = np.sign(btc['return_lag_4'] )
btc['return_lag_8'] = btc['return'].shift(8).values
btc['return_lag_8_dummy'] = np.sign(btc['return_lag_8'] )
btc['log_vol'] = np.log(btc['Volume'])
btc['ticker'] = 'btc'



# get USD index data
USDX = yf.download('DX-Y.NYB', start='2014-09-17', end='2023-02-28')
USDX = USDX.rename(columns={'Adj Close': 'USDX'})
col = ['USDX']
# merge with btc data
df = pd.concat([btc, USDX[col]], axis = 1)

# fill USDX at weekends
df['USDX'] = df['USDX'].fillna(axis=0, method="ffill")

print(df.head())

df.to_csv('btc.csv')
