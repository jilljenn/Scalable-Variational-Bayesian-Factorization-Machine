import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys


df = pd.read_csv('../vae/data/ml-100k/u.data', names=('user', 'item', 'rating', 'timestamp'), sep='\t').astype(int)
print(df.describe(), df.nunique())

try:
    df.to_csv('../vae/data/movie100k/data.csv', index=None)
except:
    print("C'est pas grave")

df_train, df_test = train_test_split(df, test_size=0.2)

with open('data/sa100k5.train_libfm', 'w') as f:
    for user, item, rating in np.array(df_train[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, 944 + item))

with open('data/sa100k5.test_libfm', 'w') as f:
    for user, item, rating in np.array(df_test[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, 944 + item))
