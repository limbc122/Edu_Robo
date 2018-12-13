import talib as ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iexfinance.stocks import get_historical_data
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def get_stats(s, n=252):
    cnt = len(s)
    wins = len(s[s>0])
    losses = len(s[s<0])
    mean_w = round(s[s>0].mean(), 3)
    mean_l = round(s[s<0].mean(), 3)
    pl = round(abs(mean_w / mean_l), 3)
    win_r = round(wins/losses, 3)
    mean_trd = round(s.mean(), 3)
    std = round(np.std(s), 3)
    max_w = round(s.max(), 3)
    min_l = round(s.min(), 3)
    sharpe = round(s.mean() / np.std(s) * np.sqrt(n), 3)
    

    print ("Trades : ", cnt,
           "\nnWins : ", wins,
           "\nnLosses : ", losses,
           "\nWin/Loss ratio : ", win_r,
           "\nProfit/Loss ratio : ", pl,
           "\nSharpe ratio : ", sharpe)


start = datetime(2015,1,1)
end = datetime(2018,1,1)

Df = get_historical_data('GLD', start=start, end=end, output_format='pandas')

split_percentage = 0.8
split = int(split_percentage * len(Df))

n = 10
Df['RSI'] = ta.RSI(np.array(Df['close'].shift(1)), timeperiod=n)
Df['SMA'] = ta.MA(Df['close'].shift(1), timeperiod=n)
Df['SAR'] = ta.SAR(np.array(Df['high'].shift(1)), np.array(Df['low'].shift(1)), 0.02, 0.2)
Df['ADX'] = ta.ADX(np.array(Df['high'].shift(1)), np.array(Df['low'].shift(1)), np.array(Df['open']), timeperiod=n)
Df['OO'] = Df['open'] - Df['open'].shift(1)
Df['OC'] = Df['open'] - Df['close'].shift(1)

Df['Ret'] = np.log(Df['close'].shift(-5) / Df['close'])
Df['Sig'] = 0
Df.loc[Df['Ret'] > Df['Ret'][:split].quantile(q=0.66), 'Sig'] = 1
Df.loc[Df['Ret'] < Df['Ret'][:split].quantile(q=0.33), 'Sig'] = -1

Df = Df.dropna()
X = Df.drop(['open','high','low','close','volume','Ret','Sig'], axis=1)

y = Df['Sig']

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

scaler = StandardScaler()
scaler.fit(X_train)

cls = SVC()
cls.fit(scaler.transform(X_train), y_train)

y_predict = cls.predict(scaler.transform(X_test))

Df['Sig'][split:] = y_predict

Df['Strategy_Return'] = Df['Sig'] * Df['Ret']

get_stats(Df.Strategy_Return[split:])
Df.Strategy_Return[split:].cumsum().plot(figsize=(10,5))
plt.title("Strategy Return")
plt.show()