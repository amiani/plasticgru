from dataclasses import dataclass
import jax.numpy as jnp
from typing import Tuple

@dataclass
class Task:
	def generate_batch(
		self,
		rng: jnp.ndarray,
		batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
		raise NotImplementedError
	
	def get_zeros(self, batch_size: int) -> jnp.ndarray:
		raise NotImplementedError