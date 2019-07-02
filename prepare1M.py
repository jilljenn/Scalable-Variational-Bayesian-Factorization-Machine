import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys


# data = '1M'
df = pd.read_csv('/Users/jilljenn/code/vae/data/movie1M/data.csv')
print(df[['user', 'item', 'rating']].head(5))
print(df[['user', 'item', 'rating']].min())
print(df[['user', 'item', 'rating']].max())
NB_USERS = 6041

train, test = train_test_split(df, test_size=0.2)
print(train.head(5))

# sys.exit(0)

with open('data/sa1M5.train_libfm', 'w') as f:
    for user, item, rating in np.array(train[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, NB_USERS + item))

# sys.exit(0)

# df = pd.read_csv('/Users/jilljenn/code/vae/data/movie1M/movie1M_test.csv')

with open('data/sa1M5.test_libfm', 'w') as f:
    for user, item, rating in np.array(test[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, NB_USERS + item))

# with open('data/sa.test_libfm') as f:
#     lines.extend(f.read().splitlines())
# print('Read test')

# random.shuffle(lines)
# print('Shuffle')

# with open('data/sa5.train_libfm', 'w') as f:
#     for line in lines[:round(0.8 * len(lines))]:
#         f.write(line + '\n')
# print('Write train')

# with open('data/sa5.test_libfm', 'w') as f:
#     for line in lines[round(0.8 * len(lines)):]:
#         f.write(line + '\n')
# print('Write test')
