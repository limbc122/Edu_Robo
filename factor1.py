import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_factor = np.random.normal(0,1,10000)
equity_names = ['Equity' + str(x) for x in range(10000)]

factor_data = pd.Series(current_factor, index=equity_names)
factor_data = pd.DataFrame(factor_data, columns=['Factor'])

factor_return = current_factor + np.random.normal(0,1,10000)

return_data = pd.Series(factor_return, index=equity_names)
return_data = pd.DataFrame(return_data, columns=['Return'])

data = return_data.join(factor_data)

ranked_data = data.sort_values(by=['Factor'])

n_baskets = int(20)
basket_returns = np.zeros(20)

for i in range(20):
    start = i * 500
    end = start + 500
    basket_returns[i] = ranked_data[start:end]['Return'].mean()

plt.bar(range(n_baskets), basket_returns)

basket_returns[19] - basket_returns[0]