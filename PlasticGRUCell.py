import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Any, Tuple
from flax.linen.recurrent import RNNCellBase
import flax.linen as nn

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any

class PlasticGRUCell(RNNCellBase):
  r"""Plastic GRU cell.

  Attributes:
    gate_fn: activation function used for gates (default: sigmoid)
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: orthogonal).
    bias_init: initializer for the bias parameters (default: zeros)
  """
  gate_fn: Callable[..., Any] = nn.activation.sigmoid
  activation_fn: Callable[..., Any] = nn.activation.tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.linear.default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.initializers.orthogonal())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

  @nn.compact
  def __call__(self, carry, inputs):
    """Gated recurrent unit (GRU) cell.

    Args:
      carry: the hidden state of the LSTM cell,
        initialized using `GRUCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h, hebb = carry

    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(nn.Dense,
                      features=hidden_features,
                      use_bias=False,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(nn.Dense,
                      features=hidden_features,
                      use_bias=True,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    r = self.gate_fn(dense_i(name='ir')(inputs) + dense_h(name='hr')(h))
    z = self.gate_fn(dense_i(name='iz')(inputs) + dense_h(name='hz')(h))

    # add bias because the linear transformations aren't directly summed.
    hn_kernel = self.param('hn_kernel',
                           self.recurrent_kernel_init,
                           (hidden_features,hidden_features))
    hn_bias = self.param('hn_bias',
                          self.bias_init,
                          (hidden_features,))
    plasticity = self.param('plasticity',
                            nn.initializers.normal(0.01),
                            (h.shape[0],h.shape[0]))
    hdotkernel = jax.lax.dot_general(h, hn_kernel + plasticity * hebb,
                                (((h.ndim - 1,), (0,)), ((), ())))
    reset_h = r * (hdotkernel + hn_bias)
    n = self.activation_fn(dense_i(name='in')(inputs) + reset_h)
    new_h = (1. - z) * n + z * h

    eta = self.param('eta', lambda _: 0.03)
    new_hebb = jnp.clip(hebb + eta * jnp.outer(h, n), -1, 1)

    return (new_h, new_hebb), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=nn.initializers.zeros):
    """initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    mem_shape = batch_dims + (size,)
    h = init_fn(rng, mem_shape)
    hebb = nn.initializers.zeros(rng, (size,size))
    return h, hebb