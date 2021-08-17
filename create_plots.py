import pickle
import matplotlib.pyplot as plt
from typing import Tuple, List
from os import walk

def plot_training_stats(
	training_stats: Tuple[List[int], List[float]],
	filename: str) -> None:

	plt.plot([i/128 for i in training_stats[0]], training_stats[1], linewidth=.2)


#exp_path = './32_48_experiments'
exp_path = './32_64_experiments'
f = []
for (dirpath, dirnames, filenames) in walk(exp_path):
    f.extend(filenames)
    break

"""
filenames = [
	'fixed experiments/BistableCell_copyfirst_300_1_8.pkl',
	'fixed experiments/PlasticBistableCell_copyfirst_300_1_8.pkl'
]
"""
for filename in f:
	with open(f'{exp_path}/{filename}', 'rb') as f:
		training_stats, test_loss = pickle.load(f)
		plt.ylim((0,1.2))
		plt.xlim((0,15625))
		plt.title(f'Copy First 32-48')
		plt.xlabel("Iterations")
		plt.ylabel("Loss")
		plot_training_stats(training_stats, filename)
		plt.legend(['Bistable Cell', 'Plastic Bistable Cell', 'PlasticGRUCell'])
		plt.savefig('plots.png', dpi=200)