import pickle
import matplotlib.pyplot as plt
from typing import Tuple, List

def plot_training_stats(
	training_stats: Tuple[List[int], List[float]],
	filename: str) -> None:
	fig = plt.figure(0, figsize=(4,3), dpi=160)
	plt.plot(training_stats[0], training_stats[1], linewidth=.1)
	plt.title("Copy First training")

filenames = [
	'fixed experiments/BistableCell_copyfirst_300_1_8.pkl',
	'fixed experiments/PlasticBistableCell_copyfirst_300_1_8.pkl'
]
for filename in filenames:
	with open(f'{filename}', 'rb') as f:
		training_stats, test_loss = pickle.load(f)
		plot_training_stats(training_stats, filename)
		plt.savefig('plots.png')