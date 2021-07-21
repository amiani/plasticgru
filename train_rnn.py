from tasks.reconstruct import ReconstructTask
from tasks.copy_first import CopyFirstTask
from tasks.task import Task
from cells import BistableCell, PlasticBistableCell, PlasticGRUCell
from RNN import RNN
import jax
import jax.numpy as jnp
import tqdm
import optax
from typing import Tuple

rng = jax.random.PRNGKey(0)
tx = optax.adam(learning_rate=1e-3)

def train(
	rng,
	task: Task,
	batch_size: int,
	hid_dim: int,
	num_iterations: int):

	carry = model.initialize_carry(rng, (batch_size,), hid_dim)
	params = model.init(rng, carry, task.get_zeros(batch_size))
	opt_state = tx.init(params)

	with tqdm.trange(num_iterations) as t:
		for epoch in t:
			rng, batch_rng, carry_rng = jax.random.split(rng, 3)
			batch = task.generate_batch(batch_rng, batch_size)
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
	batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple:

	inputs, targets = batch
	loss, grads = jax.value_and_grad(mse_loss)(params, carry, inputs, targets)
	updates, next_opt_state = tx.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, loss

def mse_loss(
	params,
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


batch_size = 128
input_dim = 128
hid_dim = 128
model = RNN(BistableCell, input_dim)
num_epochs = 50
num_iterations = int(40000*num_epochs/batch_size)
#copy_first = CopyFirstTask(300, input_dim)
reconstruct = ReconstructTask(input_dim ,3, 3, 3, 3)
params = train(rng, reconstruct, batch_size, hid_dim, num_iterations)
test_loss = test(rng, reconstruct, params, 10000, hid_dim)
print(f'Test loss: {test_loss}')