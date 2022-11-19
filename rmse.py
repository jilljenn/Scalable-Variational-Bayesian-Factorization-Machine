from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

n = sys.argv[1]


truth = pd.read_csv(f'../vae/data/movie{n}/movie{n}.test_libfm', names=('outcome', 'user', 'item'), sep=' ')
pred = pd.read_csv(f'results/mcmc_pred_{n}.csv', names=('pred',))
logs = pd.read_csv(f'results/mcmc_{n}.csv', sep='\t')
print(logs.head())

y_true = truth['outcome']
y_pred = pred['pred']
print('RMSE', mean_squared_error(y_true, y_pred) **  0.5)

plt.plot(logs['rmse_mcmc_this'], label='rmse_mcmc_this')
plt.plot(logs['rmse_mcmc_all'], label='rmse_mcmc_all')
plt.legend()
plt.show()
