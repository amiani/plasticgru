from flax import linen as nn
from typing import Type

class RNN(nn.Module):
	cell: Type[nn.GRUCell]
	out_dim: int

	@nn.compact
	def __call__(self, carry, inputs):
		rnn1 = nn.scan(self.cell,
						variable_broadcast='params',
						split_rngs={'params': False},
						in_axes=1,
						out_axes=1)()
		final_carry, y = rnn1(carry, inputs)
		output = nn.Dense(features=self.out_dim)(y)
		return final_carry, output
	
	def initialize_carry(self, rng, batch_size, hid_dim):
		return self.cell.initialize_carry(rng, batch_size, hid_dim)