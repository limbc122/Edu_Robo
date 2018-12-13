import numpy as np
import pandas as pd
from iexfinance.stocks import get_historical_data
import FinanceDataReader as fdr
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import scorer, accuracy_score


start = datetime(2013,1,1)
end = datetime(2018,1,1)
Df = get_historical_data('TSLA', start=start, end=end, output_format='pandas')

plt.figure(figsize=(20,10))
plt.title("TSLA")
Df['close'].plot()
plt.show()


krx = fdr.StockListing('KRX')