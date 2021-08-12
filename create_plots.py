import pickle
import matplotlib.pyplot as plt
from typing import Tuple, List

def plot_training_stats(
	training_stats: Tuple[List[int], List[float]],
	filename: str) -> None:
	plt.plot(training_stats[0], training_stats[1])

filenames = [
	'experiments/PlasticBistableCell_copyfirst_300_1_8.pkl',
	'experiments/BistableCell_copyfirst_300_1_128.pkl'
]
for filename in filenames:
	with open(f'{filename}', 'rb') as f:
		training_stats, test_loss = pickle.load(f)
		plot_training_stats(training_stats, filename)
		plt.savefig('plots.png')