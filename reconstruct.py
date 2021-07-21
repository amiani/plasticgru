import numpy as np
from typing import Tuple

def make_reconstruct_task(
	batch_size: int,
	dim: int,
	pres_time: int,
	pres_delay: int,
	num_patterns: int,
	num_pres_cycles: int) -> Tuple[np.ndarray, np.ndarray]:

    T = ((pres_time + pres_delay) * num_patterns) * num_pres_cycles + 3
    data = np.empty((batch_size,T,dim))
    targets = np.empty((batch_size,dim))
    rng = np.random.default_rng()

    for n in range(batch_size):
        patterns = []
        for i in range(num_patterns):
            patterns.append((rng.integers(0, 2, dim) * 2) - 1)

        for i in range(num_pres_cycles):
            rng.shuffle(patterns)
            for j in range(num_patterns):
                pattern = patterns[j]
                for a in range(pres_time):
                    t = (pres_time+pres_delay)*num_pres_cycles*i + (pres_time+pres_delay)*j + a
                    data[n,t,:] = pattern
                for a in range(pres_delay):
                    t = (pres_time+pres_delay)*num_pres_cycles*i + (pres_time+pres_delay)*j + a + pres_time
                    data[n,t,:] = np.zeros(dim)
        targets[n,:] = rng.choice(patterns)
        mask = np.random.choice(dim, (int(dim/2),), False)
        masked_pattern = targets[n,:]
        masked_pattern[mask] = 0
        data[n,-1,:] = masked_pattern
        data[n,-2,:] = masked_pattern
        data[n,-3,:] = masked_pattern
    return data, targets

print(make_reconstruct_task(1, 3, 1, 1, 2, 1))