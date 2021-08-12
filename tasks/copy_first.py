from dataclasses import dataclass
import jax
import jax.numpy as jnp
from tasks.task import Task
from typing import Tuple

@dataclass
class CopyFirstTask(Task):
	series_length: int
	input_dim: int
	
	def __init__(self, series_length, input_dim):
		self.series_length = series_length
		self.input_dim = input_dim
		self.name = f'copyfirst_{self.series_length}'

	def generate_batch(
		self,
		rng: jnp.ndarray,
		batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:

		inputs = jax.random.normal(rng, (batch_size, self.series_length, self.input_dim))
		targets = inputs[:,0,:]
		return inputs, targets
	
	def get_zeros(self, batch_size: int) -> jnp.ndarray:
		return jnp.zeros((batch_size, self.series_length, self.input_dim))