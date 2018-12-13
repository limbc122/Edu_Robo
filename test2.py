from iexfinance.stocks import get_historical_data
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import scorer, accuracy_score
#import FinanceDataReader as fdr


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

# krx = fdr.StockListing('KRX')

start = datetime(2015,1,1)
end  = datetime(2018,10,1)

Df = get_historical_data('SPY', start=start, end=end, output_format='pandas')

# plt.figure(figsize=(20,10))
# plt.title("S&P500")
# Df['close'].plot()
# plt.show()

y = np.where(Df['close'].shift(-1)>=Df['close'],1,-1)

Df['open-close'] = Df.open - Df.close
Df['high-low'] = Df.high - Df.low

X = Df[['open-close', 'high-low']]

split_percentage = 0.8
split = int(split_percentage * len(Df))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)
cls.predict(X_test)

accuracy = accuracy_score(y_test, cls.predict(X_test))

Df['Predicted_Signal'] = cls.predict(X)
Df['Return'] = np.log(Df.close.shift(-1) / Df.close) * 100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal

Df.Strategy_Return[split:].cumsum().plot(figsize=(10,5))

# plt.show()

get_stats(Df.Strategy_Return[split:])