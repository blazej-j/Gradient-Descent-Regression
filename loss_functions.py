from math_functions import DEFAULT_CONFIG

def mae(yi, ypi):
    return abs(yi - ypi)

def mse(yi, ypi):
    return (yi - ypi) ** 2

def make_huber_loss(model_fn):
    delta = DEFAULT_CONFIG[model_fn.__name__]['huber_delta']
    def huber_loss(yi, ypi):
            err = mae(yi, ypi)
            if err <= delta:
                return 0.5 * mse(yi, ypi)
            else:
                return delta * (err - 0.5 * delta)
    return huber_loss