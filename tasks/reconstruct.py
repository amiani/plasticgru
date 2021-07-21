from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from tasks.task import Task

@dataclass
class ReconstructTask(Task):
    input_dim: int
    pres_time: int
    pres_delay: int
    num_patterns: int
    num_pres_cycles: int

    def generate_batch(
        self,
        _: jnp.ndarray,
        batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:

        T = ((self.pres_time + self.pres_delay) * self.num_patterns) * self.num_pres_cycles + 3
        data = np.empty((batch_size,T,self.input_dim))
        targets = np.empty((batch_size,self.input_dim))
        rng = np.random.default_rng()

        for n in range(batch_size):
            patterns = []
            for i in range(self.num_patterns):
                patterns.append((rng.integers(0, 2, self.input_dim) * 2) - 1)

            for i in range(self.num_pres_cycles):
                rng.shuffle(patterns)
                for j in range(self.num_patterns):
                    pattern = patterns[j]
                    for a in range(self.pres_time):
                        t = (self.pres_time+self.pres_delay)*self.num_pres_cycles*i \
                            + (self.pres_time+self.pres_delay)*j + a
                        data[n,t,:] = pattern
                    for a in range(self.pres_delay):
                        t = (self.pres_time+self.pres_delay)*self.num_pres_cycles*i \
                            + (self.pres_time+self.pres_delay)*j + a + self.pres_time
                        data[n,t,:] = np.zeros(self.input_dim)
            targets[n,:] = rng.choice(patterns)
            mask = np.random.choice(self.input_dim, (int(self.input_dim/2),), False)
            masked_pattern = targets[n,:]
            masked_pattern[mask] = 0
            data[n,-1,:] = masked_pattern
            data[n,-2,:] = masked_pattern
            data[n,-3,:] = masked_pattern
        return jnp.array(data), jnp.array(targets)

    def get_zeros(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((
            batch_size,
            ((self.pres_time + self.pres_delay) * self.num_patterns) * self.num_pres_cycles + 3,
            self.input_dim))