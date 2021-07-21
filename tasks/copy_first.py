from dataclasses import dataclass
import jax
import jax.numpy as jnp
from task import Task
from typing import Tuple

@dataclass
class CopyFirstTask(Task):
	series_length: int
	input_dim: int

	def generate_batch(
		self,
		rng: jnp.ndarray,
		batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:

		inputs = jax.random.normal(rng, (batch_size, self.series_length, self.input_dim))
		targets = inputs[:,0,:]
		return inputs, targets
	
	def get_zeros(self, batch_size: int) -> jnp.ndarray:
		return jnp.zeros((batch_size, self.series_length, self.input_dim))