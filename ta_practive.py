import talib as ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iexfinance.stocks import get_historical_data
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

start = datetime(2015,1,1)
end = datetime(2018,10,1)

Df = get_historical_data('AAPL', start=start, end=end, output_format='pandas')

pd_MA = Df['close'].rolling(window=5).mean()
TA_MA = talib.MA(Df['close'], timeperiod=5)

uppper, middle, lower = talib.BBANDS(Df['close'], timeperiod=10, nbdevup=1, nbdevdn=1)

plt.figure(figsize=(15,5))
plt.plot(Df['close'], 'r--')
plt.plot(middle, 'b--')
plt.fill_between(Df.index, lower, uppper, alpha=0.3)
plt.show()