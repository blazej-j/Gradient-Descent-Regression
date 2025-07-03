import random
from math_functions import DEFAULT_CONFIG


class DataGenerator:
    def __init__(self, true_fn):
        self.true_fn = true_fn
        self.x = None
        self.y = None
        self.params = None
        self.config = DEFAULT_CONFIG[true_fn.__name__]

    def generate(self, npoints):
        self.params = [random.uniform(*rng) for rng in self.config['params_range'].values()]
        self.x = [random.uniform(*self.config['x_domain']) for _ in range(npoints)]
        self.y = [self.true_fn(xi, *self.params) + random.gauss(0, self.config['noise_std']) for xi in self.x]

        return self.x, self.y