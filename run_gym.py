from cells import PlasticGRUCell
from actor_critic_gru import ActorCriticGRU, ActorCritic
import gym
import collections
import statistics
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable
from functools import partial
#from visualize import render_episode
import tqdm
import flax
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import optax

env = gym.make("CartPole-v0")
env.seed(135)
in_dim = env.observation_space.shape
action_dim = env.action_space.n

def run_episode(rng, act, carry, max_steps: int)	\
	-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

	states = []
	actions = []
	rewards = []
	state = env.reset()[[0,2]]
	done = False
	t = 0
	while not done and t < max_steps:
		state = jnp.expand_dims(state, 0)
		states.append(state)
		carry, (_, action) = act(carry, state)
		action = int(action)
		actions.append(action)
		state, reward, done, _ = env.step(action)
		state = state[[0,2]]
		rewards.append(reward)
		t += 1
	return jnp.concatenate(states), jnp.array(actions), jnp.array(rewards)

def false_fun(stat_ret):
	(n, average, s), returns = stat_ret
	v = s*s
	for r in returns:
		n += 1.
		v = ((n - 2) / (n - 1)) * v**2 + (1/n) * (r - average)**2
		average = (1/n) * (r + (n-1) * average)
	return n, average, v**.5

eps = 1e-8
def increment_stats(stats, returns: jnp.ndarray):
	n, average, s = stats
	#print(f'n {n}, avg {average}, s {s}')
	stats = jax.lax.cond(stats[0] == 0,
				 lambda _: (len(returns)+0.0, returns.mean(), returns.std()),
				 false_fun,
				 (stats, returns))
	#print(stats)
	return stats 

def get_expected_returns(rewards: jnp.ndarray, stats, gamma: float) -> Tuple:
	returns = []
	total = 0
	for r in rewards[::-1]:
		total = r + gamma * total
		returns.insert(0, total)
	returns = jnp.array(returns)
	#stats = increment_stats(stats, returns)
	#_, average, s = stats
	average = returns.mean()
	s = returns.std()
	#print(f'Average {average}, std {s}')
	returns = (returns - average) / s
	return stats, returns

def mc_loss(params,
			rng,
			states: jnp.ndarray,
			actions: jnp.ndarray,
			returns: jnp.ndarray) -> jnp.ndarray:
	carry = model.initialize_carry(rng)
	_, (outputs, _) = model.apply(params, carry, states)
	action_outputs = outputs[:,:2]
	values = outputs[:,3]

	T = len(actions)
	logpi = jax.nn.log_softmax(action_outputs, 1)[jnp.arange(T),actions]
	advantage = returns - jax.lax.stop_gradient(values)
	actor_loss = -jnp.sum(logpi * advantage)

	critic_loss = 0.5 * jnp.mean((values - returns)**2)
	return actor_loss + critic_loss

@jax.jit
def update(params,
		   rng,
		   opt_state,
		   return_stats,
		   states: jnp.ndarray,
		   actions: jnp.ndarray,
		   rewards: jnp.ndarray,
		   gamma: float) -> Tuple:
	return_stats, returns = get_expected_returns(rewards, return_stats, gamma)
	grads = jax.grad(mc_loss)(params, rng, states, actions, returns)
	updates, next_opt_state = tx.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, return_stats

@jax.jit
def apply(params, carry, state):
	return model.apply(params, carry, state)

def train_step(params,
			   rng,
			   opt_state,
			   return_stats,
			   gamma: float,
			   max_steps: int) -> Tuple:

	k1, k2, k3 = jax.random.split(rng, 3)
	act = partial(apply, params)
	carry = model.initialize_carry(k1)
	states, actions, rewards = run_episode(k2, act, carry, max_steps)
	params, opt_state, return_stats = update(params, k3, opt_state, return_stats, states, actions, rewards, gamma)
	return rewards.sum(), params, opt_state, return_stats


min_episode_criterion = 100
max_episodes = 10000
max_steps = 200
reward_threshold = 195
gamma = 0.995
episodes_reward = collections.deque(maxlen=min_episode_criterion)

rng = jax.random.PRNGKey(98242)

hid_dim = 16
model = ActorCritic(hid_dim, 3, PlasticGRUCell)
rng, key = jax.random.split(rng)
x = jnp.zeros((1,2))
carry = model.initialize_carry(key)
params = model.init(key, carry, x)
tx = optax.adam(learning_rate=1e-3)
opt_state = tx.init(params)
return_stats = (0, 0, 0)

prefix = f'plastic{hid_dim}_'
load = True
if load:
	params, opt_state, rewards_list = restore_checkpoint('checkpoints', (params, opt_state, [0]*100), prefix=prefix)
	episodes_reward = collections.deque(rewards_list, maxlen=min_episode_criterion)

train = True
if train:
	with tqdm.trange(max_episodes) as t:
		for ep in t:
			rng, key = jax.random.split(rng)
			ep_reward, params, opt_state, return_stats = train_step(params, key, opt_state, return_stats, gamma, max_steps)
			episodes_reward.append(int(ep_reward))
			running_reward = statistics.mean(episodes_reward)

			t.set_description(f'Episode {ep}')
			t.set_postfix(episode_reward=ep_reward, running_reward=running_reward)

			if ep % 100 == 0:
				save_checkpoint('checkpoints',
								(params, opt_state, list(episodes_reward)),
								running_reward,
								prefix,
								keep=3,
								overwrite=True)

			if running_reward > reward_threshold and ep >= min_episode_criterion:
				save_checkpoint('checkpoints',
								(params, opt_state, list(episodes_reward)),
								running_reward,
								prefix,
								keep=3,
								overwrite=True)
				break
	print(f'\nSolved at episode {ep}: average reward: {running_reward:.2f}!')

save_gif = True
if save_gif:
	env.seed(101)
	# Save GIF image
	act = partial(apply, params)
	#carry = model.initialize_carry(rng)
	images = render_episode(env, act, carry, max_steps)
	image_file = 'cartpole-v0.gif'
	# loop=0: loop forever, duration=1: play each frame for 1ms
	images[0].save(
		image_file, save_all=True, append_images=images[1:], loop=0, duration=1)