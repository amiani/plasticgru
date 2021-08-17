import matplotlib.pyplot as plt
from flax.core.frozen_dict import FrozenDict
from tasks.reconstruct import ReconstructTask
from tasks.copy_first import CopyFirstTask
from tasks.task import Task
from cells import BistableCell, PlasticBistableCell, PlasticGRUCell, PlasticBistableCell2
from RNN import RNN
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import List, Tuple
import tqdm
import pickle

rng = jax.random.PRNGKey(0)
tx = optax.adam(learning_rate=1e-3)

def save_training_stats(
	training_stats: Tuple[List[int], List[float]],
	test_loss: float,
	name: str) -> None:
	with open(f'{name}.pkl', 'wb') as f:
		pickle.dump((training_stats, test_loss), f)

def train(
	rng,
	task: Task,
	batch_size: int,
	hid_dim: int,
	num_iterations: int) -> Tuple[FrozenDict, Tuple[List, List]]:

	carry = model.initialize_carry(rng, (batch_size,), hid_dim)
	params = model.init(rng, carry, task.get_zeros(batch_size))
	opt_state = tx.init(params)
	training_stats = ([], [])

	with tqdm.trange(num_iterations) as t:
		for epoch in t:
			rng, batch_rng, carry_rng = jax.random.split(rng, 3)
			batch = task.generate_batch(batch_rng, batch_size)
			carry = model.initialize_carry(carry_rng, (batch_size,), hid_dim)
			params, opt_state, loss = update(params, opt_state, carry, batch)
			training_stats[0].append(epoch*batch_size)
			training_stats[1].append(loss)

			if loss < 0.005:
				break

			t.set_description(f'Epoch {epoch}')
			t.set_postfix(loss=loss)
	
	return params, training_stats

@jax.jit
def update(
	params,
	opt_state,
	carry: jnp.ndarray,
	batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple:

	inputs, targets = batch
	loss, grads = jax.value_and_grad(mse_loss)(params, carry, inputs, targets)
	updates, next_opt_state = tx.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, loss

def mse_loss(
	params: FrozenDict,
	carry: jnp.ndarray,
	inputs: jnp.ndarray,
	targets: jnp.ndarray) -> jnp.ndarray:

	carry, output = model.apply(params, carry, inputs)
	return jnp.mean((output[:,-1,:] - targets)**2)

def test(
	rng,
	task: Task,
	params,
	batch_size: int,
	hid_dim: int) -> jnp.ndarray:

	rng, batch_rng = jax.random.split(rng)
	inputs, targets = task.generate_batch(batch_rng, batch_size)
	carry = model.initialize_carry(rng, (batch_size,), hid_dim)
	return mse_loss(params, carry, inputs, targets)

def run_experiment(
	cell: nn.recurrent.GRUCell,
	task: Task,
	batch_size = 128,
	input_dim = 1,
	hid_dim = 2,
	num_epochs = 50) -> None:

	num_iterations = int(40000*num_epochs/batch_size)
	params, training_stats = train(rng, task, batch_size, hid_dim, num_iterations)
	test_loss = test(rng, task, params, 10000, hid_dim)
	print(f'Test loss: {test_loss}')
	save_training_stats(training_stats, test_loss, f'{cell.__name__}_{task.name}_{input_dim}_{hid_dim}')

batch_size = 128
input_dim = 32
hid_dim = 48
num_epochs = 50
cells = [BistableCell]
copy_first = CopyFirstTask(300, input_dim)
reconstruct = ReconstructTask(input_dim, 3, 3, 3, 3)

for cell in cells:
	model = RNN(cell, input_dim)
	run_experiment(
		cell,
		copy_first,
		batch_size,
		input_dim,
		hid_dim,
		num_epochs
	)