import numpy as np

# --- Stochastic Gradient Descent (SGD) ---
class sgd:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        print(f"Learning rate: {self.learning_rate}")
        self.weight_decay = weight_decay

    def update(self, W, b, dW, db):
        W -= self.learning_rate * (dW + self.weight_decay * W)
        b -= self.learning_rate * db

# --- Momentum-based SGD ---
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None

    def update(self, W, b, dW, db):
        if self.vW is None:
            self.vW = np.zeros_like(W)
            self.vb = np.zeros_like(b)

        self.vW = self.momentum * self.vW - self.learning_rate * (dW + self.weight_decay * W)  # Updated (Line 26)
        self.vb = self.momentum * self.vb - self.learning_rate * db

        W += self.vW
        b += self.vb

# --- Nesterov Accelerated Gradient (NAG) ---
class NAG:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None

    def update(self, W, b, dW, db):
        if self.vW is None:
            self.vW = np.zeros_like(W)
            self.vb = np.zeros_like(b)

        W_lookahead = W + self.momentum * self.vW - self.weight_decay * W  # Updated (Line 47)
        b_lookahead = b + self.momentum * self.vb

        self.vW = self.momentum * self.vW - self.learning_rate * (dW + self.weight_decay * W_lookahead)
        self.vb = self.momentum * self.vb - self.learning_rate * db

        W += self.vW
        b += self.vb

# --- RMSprop Optimizer ---
class RMSprop:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.sW = None
        self.sb = None

    def update(self, W, b, dW, db):
        if self.sW is None:
            self.sW = np.zeros_like(W)
            self.sb = np.zeros_like(b)

        self.sW = self.beta * self.sW + (1 - self.beta) * dW**2
        self.sb = self.beta * self.sb + (1 - self.beta) * db**2

        W -= self.learning_rate * dW / (np.sqrt(self.sW) + self.epsilon) + self.weight_decay * W
        b -= self.learning_rate * db / (np.sqrt(self.sb) + self.epsilon)

# --- Adam Optimizer ---
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.mW = []
        self.vW = []
        self.mb = []
        self.vb = []
        self.t = 0

    def update(self, W, b, dW, db,layer_idx):
        
        while len(self.mW) <= layer_idx:
            self.mW.append(np.zeros_like(W))
            self.vW.append(np.zeros_like(W))
            self.mb.append(np.zeros_like(b))
            self.vb.append(np.zeros_like(b))
        # Increment time step
        self.t += 1  

        self.mW[layer_idx] = self.beta1 * self.mW[layer_idx] + (1 - self.beta1) * dW
        self.mb[layer_idx] = self.beta1 * self.mb[layer_idx] + (1 - self.beta1) * db

        # Compute biased second raw moment estimate
        self.vW[layer_idx] = self.beta2 * self.vW[layer_idx] + (1 - self.beta2) * (dW ** 2)
        self.vb[layer_idx] = self.beta2 * self.vb[layer_idx] + (1 - self.beta2) * (db ** 2)

        mW_hat = self.mW[layer_idx] / (1 - self.beta1**self.t)  # Updated (Line 93)
        vW_hat = self.vW[layer_idx] / (1 - self.beta2**self.t)  # Updated (Line 97)

        mb_hat = self.mb[layer_idx] / (1 - self.beta1 ** self.t)
        vb_hat = self.vb[layer_idx] / (1 - self.beta2 ** self.t)

        W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.epsilon) + self.weight_decay * W
        b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)

# --- Nadam Optimizer ---
class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.mW = None
        self.vW = None
        self.mb = None
        self.vb = None
        self.t = 0

    def update(self, W, b, dW, db):
        if self.mW is None:
            self.mW = np.zeros_like(W)
            self.vW = np.zeros_like(W)
            self.mb = np.zeros_like(b)
            self.vb = np.zeros_like(b)

        self.t += 1

        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dW**2)
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (db**2)

        mW_hat = self.mW / (1 - self.beta1**self.t)  # Updated (Line 129)
        vW_hat = self.vW / (1 - self.beta2**self.t)  # Updated (Line 133)

        W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.epsilon) + self.weight_decay * W
        b -= self.learning_rate * self.mb / (np.sqrt(self.vb) + self.epsilon)

# --- Function to Retrieve Optimizer Class ---
def get_optimizer(name, model, args):
    optimizers = {
        "sgd": sgd(args.learning_rate, args.weight_decay),
        "momentum": Momentum(args.learning_rate, args.momentum, args.weight_decay),
        "nag": NAG(args.learning_rate, args.momentum, args.weight_decay),
        "rmsprop": RMSprop(args.learning_rate, args.beta, args.epsilon, args.weight_decay),
        "adam": Adam(args.learning_rate, args.beta1, args.beta2, args.epsilon, args.weight_decay),
        "nadam": Nadam(args.learning_rate, args.beta1, args.beta2, args.epsilon, args.weight_decay)
    }
    return optimizers.get(name, sgd(args.learning_rate))
