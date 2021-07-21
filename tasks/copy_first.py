import jax
import jax.numpy as jnp
from typing import Dict

def generate_copy_first_batch(rng: jnp.ndarray, config: Dict) -> jnp.ndarray:
	batch_size = config["batch_size"]
	series_length = config["series_length"]
	input_dim = config["input_dim"]
	return jax.random.normal(rng, (batch_size, series_length, input_dim))

copy_first_config = {
	'name': 'copy first',
	'generate_batch': generate_copy_first_batch,
	'batch_size': 128,
	'series_length': 300,
	'input_dim': 128,
}