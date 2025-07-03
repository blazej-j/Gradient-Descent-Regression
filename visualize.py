import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

class Visualizer:
    def __init__(self,generator,model):
        self.generator = generator
        self.model = model

    def plot_points(self):
        x = self.generator.x
        y = self.generator.y

        plt.scatter(x,y, alpha=0.7,s=20)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{self.model.model_fn.__name__} fit")

    def _plot_model(self, params, color, **kwargs):
        x_sorted = sorted(self.generator.x)
        y_pred = [self.model.model_fn(xi, *params) for xi in x_sorted]
        plt.plot(x_sorted, y_pred, color=color, **kwargs)

    def plot_initial_model(self):
        initial_params = self.model.params_history[0]
        self._plot_model(params=initial_params, color="red", linewidth= 1.5)

    def plot_model_progression(self):
        progression_params = self.model.params_history[1:-1]
        for pp in progression_params:
            self._plot_model(params=pp, color="blue", linewidth= 0.25, alpha = 0.3)

    def plot_final_model(self):
        final_params = self.model.params_history[-1]
        self._plot_model(params=final_params, color="green", linewidth= 1.5)


    def plot_training_overview(self):
        self.plot_points()
        self.plot_initial_model()
        self.plot_model_progression()
        self.plot_final_model()
        plt.show()
    
    def plot_error(self):
        x = list(range(len(self.model.loss_history)))
        y = self.model.loss_history

        plt.plot(x, y, color='blue', label='error change')
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.title(f"loss_history")
        plt.show()