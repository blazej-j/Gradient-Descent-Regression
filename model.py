import random
from math_functions import DEFAULT_CONFIG


class RegressionModel:
    def __init__(self,model_fn, loss_fn):
        self.model_fn = model_fn
        self.config = DEFAULT_CONFIG[model_fn.__name__]
        self.params = [random.uniform(*rng) for rng in self.config['params_range'].values()]
        self.loss_history = []
        self.params_history = [self.params]
        self.loss_fn = loss_fn
        
    def predict(self,x, params):
        return [self.model_fn(xi, *params) for xi in x]

    def loss(self, x, y, params):
        ypred = self.predict(x,params)
        errors = [self.loss_fn(yi,ypi) for yi, ypi in zip(y, ypred)]
        
        return sum(errors) / len(errors)

    def score(self, x, y):
        y_pred = self.predict(x, self.params)
        y_mean = sum(y) / len(y)
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        return 1 - ss_res / ss_tot

    def compute_gradients(self, x, y):
        delta = self.config['grad_delta']
        base_error = self.loss(x, y, self.params)

        grads = []
        for i in range(len(self.params)):
            params_copy = list(self.params)
            params_copy[i] += delta

            new_error = self.loss(x, y, params_copy)
            grad = (new_error - base_error) / delta
            grads.append(grad)  
        
        return grads
    
    def fit(self, x, y, epochs):
        lr = self.config['lr']
        for e in range(epochs):

            grads = self.compute_gradients(x, y)
            
            self.params = [p - lr * g for p, g in zip(self.params, grads)]
            self.params_history.append(list(self.params))

            error = self.loss(x, y, self.params)
            self.loss_history.append(error)
            
        
        return self.loss_history, self.params_history