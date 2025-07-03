import math

DEFAULT_CONFIG = {
    'linear_fn': {
        'params_range': {
            'a': (-15, 15),
            'b': (-15, 15),
        },
        'x_domain': (-8, 8),
        'noise_std': 2,
        'lr': 0.05,
        'grad_delta': 0.05,
        'huber_delta': 0.1
    },

    'square_fn': {
        'params_range': {
            'a': (-2, 2),
        },
        'x_domain': (-6, 6),
        'noise_std': 0.5,
        'lr': 0.01,
        'grad_delta': 0.03,
        'huber_delta': 0.1
    },

    'cubic_fn': {
        'params_range': {
            'a': (-1, 1),
            'b': (-1, 1),
            'c': (-1, 1),
            'd': (-1, 1),
        },
        'x_domain': (-4, 4),
        'noise_std': 0.5,
        'lr': 0.0008,
        'grad_delta': 0.03,
        'huber_delta': 0.1
    },

    'exp_fn': {
        'params_range': {
            'a': (1, 1.2),
            'b': (0.1, 1.0),
        },
        'x_domain': (-1, 1),
        'noise_std': 0.05,
        'lr': 0.01,
        'grad_delta': 0.02,
        'huber_delta': 0.05
    },

    'log_fn': {
        'params_range': {
            'a': (0.5, 2.0),
            'b': (1, 3)
        },
        'x_domain': (0.2, 4),
        'noise_std': 0.15,
        'lr': 0.02,
        'grad_delta': 0.015,
        'huber_delta': 0.05
    },
        'sin_fn': {
        'params_range': {
            'a': (0.5, 2.0),
            'b': (0.5, 3.0),
            'c': (-math.pi, math.pi)
        },
        'x_domain': (-2*math.pi, 2*math.pi),
        'noise_std': 0.1,
        'lr': 0.2,
        'grad_delta': 0.02,
        'huber_delta': 0.05
    },

    'tan_fn': {
        'params_range': {
            'a': (0.5, 1.5),
            'b': (0.1, 0.5),
            'c': (-0.5, 0.5)
        },
        'x_domain': (-1, 1),
        'noise_std': 0.1,
        'lr': 0.015,
        'grad_delta': 0.01,
        'huber_delta': 0.05
    }
}

def linear_fn(x, a, b):
    return a * x + b

def square_fn(x,a):
    return a * x**2

def cubic_fn(x,a,b,c,d):
    return a* x**3 + b * x**2 + c*x + d

def exp_fn(x, a, b):
    return a * math.exp(b * x)

def log_fn(x, a, b):
    return a * math.log(x + b)

def sin_fn(x, a, b, c):
    return a * math.sin(b * x + c)

def tan_fn(x, a, b, c):
    return a * math.tan(b * x + c)