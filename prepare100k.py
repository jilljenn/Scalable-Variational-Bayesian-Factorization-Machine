import random
import pandas as pd
import numpy as np
import sys


df = pd.read_csv('/Users/jilljenn/code/vae/data/movie100k/movie100k_train.csv')
# print(df[['user', 'item', 'rating']].head(5))
# print(df[['user', 'item', 'rating']].min())
# print(df[['user', 'item', 'rating']].max())

with open('data/sa100k5.train_libfm', 'w') as f:
    for user, item, rating in np.array(df[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, 944 + item))

df = pd.read_csv('/Users/jilljenn/code/vae/data/movie100k/movie100k_test.csv')

with open('data/sa100k5.test_libfm', 'w') as f:
    for user, item, rating in np.array(df[['user', 'item', 'rating']]):
        f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, 944 + item))

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
