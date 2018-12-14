import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels
from statsmodels.tsa.stattools import coint
from iexfinance.stocks import get_historical_data
from datetime import datetime

start = datetime(2013,1,1)
end = datetime(2018,1,1)



MSFT = get_historical_data("MSFT", start=start, end=end, output_format='pandas')
ADBE = get_historical_data("ADBE", start=start, end=end, output_format='pandas')

data = pd.DataFrame([ADBE.close, MSFT.close]).T
data.columns = ['ADBE', 'MSFT']

S1 = data['ADBE']
S2 = data['MSFT']

ratio = S1/S2
# ratio.plot(figsize=(15,7))
# plt.axhline(ratio.mean())
# plt.show()

def zscore(series):
    return (series-series.mean())/np.std(series)

# zscore(ratio).plot(figsize=(15,7))
# plt.axhline(zscore(ratio).mean(), color='black')
# plt.axhline(1.0, color='r', linestyle='--')
# plt.axhline(-1.0, color='green', linestyle='--')
# plt.legend(['Z-score', 'Mean', '+1', '-1'])
# plt.show()

train = ratio[:800]
test = ratio[800:]

ratio_ma5 = train.rolling(window=5).mean()
ratio_ma60 = train.rolling(window=60).mean()
std_60 = train.rolling(window=60).std()
zscore = (ratio_ma5 - ratio_ma60) / std_60

zscore.plot(figsize=(15,7))
plt.axhline(0, color='black')
plt.axhline(1.0, color='r', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Z-score', 'Mean', '+1', '-1'])
plt.show()

buy = train.copy()
sell = train.copy()
buy[zscore > -1] = 0
sell[zscore < 1] = 0
train[60:].plot()
