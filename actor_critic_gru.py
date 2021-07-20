from flax.linen.recurrent import GRUCell
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Type

class ActorCriticGRU(nn.Module):
	hid_dim: int
	out_dim: int

	def setup(self):
		self.gru = nn.scan(nn.GRUCell,
						   variable_broadcast='params',
						   split_rngs={'params': False})()
		self.dense = nn.Dense(features=self.out_dim)

	def __call__(self, carry, inputs):
		rng, h = carry
		final_h, y = self.gru(h, inputs)
		output = self.dense(y)
		rng, act_rng = jax.random.split(rng)
		action = jax.random.categorical(act_rng, output[:,:2])
		return (rng, final_h), (output, action)
	
	def initialize_carry(self, rng):
		h = nn.GRUCell.initialize_carry(rng, (), self.hid_dim)
		return rng, h

class ActorCritic(nn.Module):
	hid_dim: int
	out_dim: int
	cell: Type[GRUCell]

	@nn.compact
	def __call__(self, carry, inputs):
		rng, *state = carry
		state = tuple(state)
		rnn = nn.scan(self.cell,
						variable_broadcast='params',
						split_rngs={'params': False})()
		final_state, y = rnn(state, inputs)
		output = nn.Dense(features=self.out_dim)(y)
		rng, act_rng = jax.random.split(rng)
		action = jax.random.categorical(act_rng, output[:,:2])
		return (rng, *final_state), (output, action)
	
	def initialize_carry(self, rng):
		state = self.cell.initialize_carry(rng, (), self.hid_dim)
		return rng, *state