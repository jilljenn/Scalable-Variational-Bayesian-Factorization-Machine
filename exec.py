import sys
from pathlib import Path
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Run VFM baselines')
parser.add_argument('data', type=str, nargs='?', default='fraction')
# parser.add_argument('--regression', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--classification', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--d', type=int, nargs='?', default=3)
parser.add_argument('--nb_batches', type=str, nargs='?', default='1')
parser.add_argument('--max_epochs', type=str, nargs='?', default='200')
parser.add_argument('--method', type=str, nargs='?', default='mcmc')
options = parser.parse_args()

DATA = sys.argv[1]
is_classification = DATA.endswith('binary') or DATA.endswith('100') or DATA.endswith('1000')
PATH = Path(f'../vae/data/{DATA}/')
EMBEDDING_SIZE = options.d
RESULTS_PATH = Path('results')
METHOD = options.method
MODEL_NAME = f'{METHOD}_{DATA}_{EMBEDDING_SIZE}'

COMMAND = [
	'time', './bin/libFM',
	'-task', 'c' if is_classification else 'r',
	'-train', PATH / f'{DATA}.trainval_libfm',
	'-test', PATH / f'{DATA}.test_libfm',
	'-dim', f'1,1,{EMBEDDING_SIZE}',
	'-method', METHOD,
	'-batch', options.nb_batches,
	'-iter', options.max_epochs,
	'-rlog', RESULTS_PATH / f'{MODEL_NAME}.csv',
	'-out', RESULTS_PATH / f'pred_{MODEL_NAME}.csv'
] + (['-verbosity', '1'] if METHOD != 'vb_online' else [])
print(' '.join(map(str, COMMAND)))
for line in COMMAND:
	print(str(line))
subprocess.Popen(COMMAND)
