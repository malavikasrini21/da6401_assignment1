import numpy as np

# --- Identity Activation ---
class Identity:
    def __call__(self, x):
        self.x = x
        return x

    def diff(self, prev_grad):
        return prev_grad

# --- Sigmoid Activation ---
class Sigmoid:
    def __call__(self, x):
        self.x = x
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def diff(self, prev_grad):
        return prev_grad * (self.out * (1 - self.out))

# --- Tanh Activation ---
class Tanh:
    def __call__(self, x):
        self.x = x
        self.out = np.tanh(x)
        return self.out

    def diff(self, prev_grad):
        return prev_grad * (1 - self.out ** 2)

# --- ReLU Activation ---
class ReLU:
    def __call__(self, x):
        self.x = x
        self.out = np.maximum(0, x)
        return self.out

    def diff(self, prev_grad):
        return prev_grad * (self.out > 0).astype(float)  # Used self.out instead of self.x

# --- Softmax Activation (for output layer) ---
class Softmax:
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def diff(self, prev_grad):
        return prev_grad  # Softmax derivative is handled with cross-entropy

# --- Function to Retrieve Activation Class ---
def get_activation(name):
    activations = {
        "identity": Identity(),
        "sigmoid": Sigmoid(),
        "tanh": Tanh(),
        "ReLU": ReLU(),
        "softmax": Softmax()
    }
    return activations.get(name, Sigmoid())  # Default to Sigmoid
