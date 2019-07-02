import random


with open('data/sa.train_libfm') as f:
    lines = f.read().splitlines()
print('Read train')

with open('data/sa.test_libfm') as f:
    lines.extend(f.read().splitlines())
print('Read test')

random.shuffle(lines)
print('Shuffle')

with open('data/sa5.train_libfm', 'w') as f:
    for line in lines[:round(0.8 * len(lines))]:
        f.write(line + '\n')
print('Write train')

with open('data/sa5.test_libfm', 'w') as f:
    for line in lines[round(0.8 * len(lines)):]:
        f.write(line + '\n')
print('Write test')
