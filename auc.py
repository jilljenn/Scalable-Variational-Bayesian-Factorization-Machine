from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import pandas as pd
import numpy as np
import sys

data = sys.argv[1]
method = sys.argv[2]
embedding_size = sys.argv[3]

truth = pd.read_csv(f'../vae/data/{data}/{data}.test_libfm', names=('outcome', 'user', 'item'), sep=' ')
pred = pd.read_csv(f'results/pred_{method}_{data}_{embedding_size}.csv', names=('pred',))
y_true = truth['outcome']
y_pred = pred['pred']
print('ACC', accuracy_score(y_true, np.round(y_pred)))
print('AUC', roc_auc_score(y_true, y_pred))
print('MAP', average_precision_score(y_true, y_pred))
