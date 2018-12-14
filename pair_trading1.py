import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels
from statsmodels.tsa.stattools import coint

Xreturn = np.random.normal(0,1,100)
X = pd.Series(np.cumsum(Xreturn), name='X') + 50

noise = np.random.normal(0,1,100)
Y = X + 5 + noise
Y.name = 'Y'
pd.concat([X,Y], axis=1).plot(figsize=(15,7))
# plt.show()

(Y/X).plot(figsize=(15,7))
plt.axhline((Y/X).mean(), color='r', linestyle='--')
plt.legend(['Price Ratio', 'Mean'])
plt.show()