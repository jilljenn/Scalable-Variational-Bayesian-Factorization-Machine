import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys


df = pd.read_csv('../vae/data/ml-10m/ml-10M100K/ratings.dat', names=('user', 'item', 'rating', 'timestamp'), sep='::').astype(int)
print(df.describe(), df.nunique())
NB_USERS = 69878

train, test = train_test_split(df, test_size=0.2)
print(train.head(5))

with open('data/sa10M5.train_libfm', 'w') as f:
    for user, item, rating in np.array(train[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, NB_USERS + item))

with open('data/sa10M5.test_libfm', 'w') as f:
    for user, item, rating in np.array(test[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, NB_USERS + item))
