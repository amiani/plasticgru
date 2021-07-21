from tasks.copy_first import copy_first_config
from cells import BistableCell, PlasticBistableCell, PlasticGRUCell
from flax import linen as nn
from RNN import RNN
import jax
import jax.numpy as jnp
import tqdm
import optax
from typing import Callable, Dict

rng = jax.random.PRNGKey(0)
tx = optax.adam(learning_rate=1e-3)

def train(
	rng,
	task_config: Dict,
	input_dim: int,
	hid_dim: int,
	series_length: int,
	num_iterations: int):

	batch_size = task_config['batch_size']
	carry = model.initialize_carry(rng, (batch_size,), hid_dim)
	x = jnp.zeros((batch_size, series_length, input_dim))
	params = model.init(rng, carry, x)
	opt_state = tx.init(params)

	generate_batch = task_config['generate_batch']
	with tqdm.trange(num_iterations) as t:
		for epoch in t:
			rng, batch_rng, carry_rng = jax.random.split(rng, 3)
			#batch = jax.random.normal(batch_rng, (batch_size, series_length, input_dim))
			batch = generate_batch(batch_rng, task_config)
			carry = model.initialize_carry(carry_rng, (batch_size,), hid_dim)
			params, opt_state, loss = update(params, opt_state, carry, batch)

			t.set_description(f'Epoch {epoch}')
			t.set_postfix(loss=loss)
	
	return params

@jax.jit
def update(
	params,
	opt_state,
	carry: jnp.ndarray,
	batch: jnp.ndarray):

	loss, grads = jax.value_and_grad(mse_loss)(params, carry, batch)
	updates, next_opt_state = tx.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, loss

def mse_loss(
	params,
	carry: jnp.ndarray,
	batch: jnp.ndarray) -> jnp.ndarray:

	carry, output = model.apply(params, carry, batch)
	return jnp.mean((output[:,-1,:] - batch[:,0,:])**2)

def test(
	rng,
	params,
	batch_size: int,
	input_dim: int,
	hid_dim: int,
	series_length: int) -> jnp.ndarray:

	rng, batch_rng = jax.random.split(rng)
	batch = jax.random.normal(batch_rng, (batch_size, series_length, input_dim))
	carry = model.initialize_carry(rng, (batch_size,), hid_dim)
	return mse_loss(params, carry, batch)


batch_size = 128
input_dim = 128
hid_dim = 128
model = RNN(BistableCell, input_dim)
series_length = 300
num_epochs = 50
num_iterations = int(40000*num_epochs/batch_size)
params = train(rng, copy_first_config, input_dim, hid_dim, series_length, num_iterations)
test_loss = test(rng, params, 10000, input_dim, hid_dim, series_length)
print(f'Test loss: {test_loss}')