import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys


DATA = sys.argv[1]
PATH = Path('../vae/data')
if DATA == '100k':
    from_ = '/ml-100k/u.data'
    SEP = '\t'
    to_vfm = 'movie100k/'
    to_ovbfm = 'sa100k5'
elif DATA == '1M':
    from_ = 'ml-1m/ratings.dat'
    SEP = '::'
    to_vfm = 'movie1M'
    to_ovbfm = 'sa1M5'


df = pd.read_csv(PATH / from_, names=('user', 'item', 'rating', 'timestamp'), sep=SEP, engine='python').astype(int)
NB_USERS_MAX = 1 + df['user'].nunique()
df['item'] += NB_USERS_MAX
i_train, i_test = train_test_split(df.reset_index()[['index']], test_size=0.2, shuffle=True)

df.to_csv(PATH / to_vfm / 'data.csv', index=None)
i_train.to_csv(PATH / to_vfm / 'trainval.csv', index=None)
i_test.to_csv(PATH / to_vfm / 'test.csv', index=None)

with open(f'data/{to_ovbfm}.train_libfm', 'w') as f:
    for user, item, rating in np.array(df.loc[i_train['index'], ['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, item))

with open(f'data/{to_ovbfm}.test_libfm', 'w') as f:
    for user, item, rating in np.array(df.loc[i_test['index'], ['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, item))
