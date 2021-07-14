from typing import Callable, Tuple
import gym
import jax.numpy as jnp

"""
# Install additional packages for visualization
sudo apt-get install -y xvfb python-opengl > /dev/null 2>&1
pip install pyvirtualdisplay > /dev/null 2>&1
pip install git+https://github.com/tensorflow/docs > /dev/null 2>&1
"""
# Render an episode and save as a GIF file

#from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display


display = Display(visible=False, size=(400, 300))
display.start()


def render_episode(env: gym.Env, act: Callable, carry: Tuple, max_steps: int): 
  state = env.reset()[[0,2]]
  screen = env.render(mode='rgb_array')
  im = Image.fromarray(screen)

  images = [im]
  
  #state = env.reset()
  rewards = []
  for i in range(1, max_steps + 1):
    state = jnp.expand_dims(state, 0)
    _, (outputs, action) = act(carry, state)
    action = int(jnp.argmax(outputs[:,:2]))

    state, reward, done, _ = env.step(action)
    rewards.append(reward)
    state = state[[0,2]]

    # Render screen every 10 steps
    if i % 10 == 0:
      screen = env.render(mode='rgb_array')
      images.append(Image.fromarray(screen))
  
    if done:
      break
  
  print(sum(rewards))
  return images