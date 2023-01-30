from sklearn.metrics import (
	roc_auc_score, accuracy_score, average_precision_score, mean_squared_error)
import pandas as pd
import numpy as np
import sys


data = sys.argv[1]
methods = sys.argv[2]
embedding_size = sys.argv[3]


truth = pd.read_csv(f'../vae/data/{data}/{data}.test_libfm',
					names=('outcome', 'user', 'item', 'extra'), sep=' ')
for method in methods.split(','):
	pred = pd.read_csv(f'results/pred_{method}_{data}_{embedding_size}.csv',
					   names=('pred',))
	print(truth)
	y_true = truth['outcome']
	y_pred = pred['pred']
	try:
		print(f'{method} ACC', accuracy_score(y_true, np.round(y_pred)))
		print(f'{method} AUC', roc_auc_score(y_true, y_pred))
		print(f'{method} MAP', average_precision_score(y_true, y_pred))
	except Exception as e:
		print(e)
	print(f'{method} RMSE', mean_squared_error(y_true, y_pred) **  0.5)
