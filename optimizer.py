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
        self.vW = []
        self.vb = []

    def update(self, W, b, dW, db,layer_idx):
        if len(self.vW) <= layer_idx:
            self.vW.append(np.zeros_like(W))
            self.vb.append(np.zeros_like(b))

        self.vW[layer_idx] = self.momentum * self.vW[layer_idx] - self.learning_rate * (dW + self.weight_decay * W)
        self.vb[layer_idx] = self.momentum * self.vb[layer_idx] - self.learning_rate * db

        W += self.vW[layer_idx]
        b += self.vb[layer_idx]

# --- Nesterov Accelerated Gradient (NAG) ---
class NAG:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.vW = []
        self.vb = []

    def update(self, W, b, dW, db,layer_idx):
        if len(self.vW) <= layer_idx:
            self.vW.append(np.zeros_like(W))
            self.vb.append(np.zeros_like(b))

        W_lookahead = W + self.momentum * self.vW[layer_idx] - self.weight_decay * W  
        b_lookahead = b + self.momentum * self.vb[layer_idx]

        self.vW[layer_idx] = self.momentum * self.vW[layer_idx] - self.learning_rate * (dW + self.weight_decay * W_lookahead)
        self.vb[layer_idx] = self.momentum * self.vb[layer_idx] - self.learning_rate * db

        W += self.vW[layer_idx]
        b += self.vb[layer_idx]

# --- RMSprop Optimizer ---
class RMSprop:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.sW = []
        self.sb = []

    def update(self, W, b, dW, db,layer_idx):
        if len(self.sW) <= layer_idx:
            self.sW.append(np.zeros_like(W))
            self.sb.append(np.zeros_like(b))
        
        self.sW[layer_idx] = self.beta * self.sW[layer_idx] + (1 - self.beta) * dW**2
        self.sb[layer_idx] = self.beta * self.sb[layer_idx] + (1 - self.beta) * db**2

        W -= self.learning_rate * dW / (np.sqrt(self.sW[layer_idx]) + self.epsilon) + self.weight_decay * W
        b -= self.learning_rate * db / (np.sqrt(self.sb[layer_idx]) + self.epsilon)

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
        self.mW = []
        self.vW = []
        self.mb = []
        self.vb = []
        self.t = 0

    def update(self, W, b, dW, db,layer_idx):
        if len(self.mW) <= layer_idx:
            self.mW.append(np.zeros_like(W))
            self.vW.append(np.zeros_like(W))
            self.mb.append(np.zeros_like(b))
            self.vb.append(np.zeros_like(b))
        self.t += 1

        self.mW[layer_idx] = self.beta1 * self.mW[layer_idx] + (1 - self.beta1) * dW
        self.vW[layer_idx] = self.beta2 * self.vW[layer_idx] + (1 - self.beta2) * (dW**2)
        self.mb[layer_idx] = self.beta1 * self.mb[layer_idx] + (1 - self.beta1) * db
        self.vb[layer_idx] = self.beta2 * self.vb[layer_idx] + (1 - self.beta2) * (db**2)

        mW_hat = self.mW[layer_idx] / (1 - self.beta1**self.t) 
        vW_hat = self.vW[layer_idx] / (1 - self.beta2**self.t)

        m_nadam_w = (self.beta1 * mW_hat + (1 - self.beta1) * dW / (1 - self.beta1 ** self.t))

        W -= self.learning_rate * m_nadam_w / (np.sqrt(vW_hat) + self.epsilon) + self.weight_decay * W

        mb_hat = self.mb[layer_idx] / (1 - self.beta1 ** self.t)
        vb_hat = self.vb[layer_idx] / (1 - self.beta2 ** self.t)

        m_nadam_b = (self.beta1 * mb_hat + (1 - self.beta1) * db / (1 - self.beta1 ** self.t))

        b -= self.learning_rate * m_nadam_b / (np.sqrt(vb_hat)) + self.epsilon

# --- Function to Retrieve Optimizer Class ---
def get_optimizer(name, model, config,args):
    if config is None:
        config = args
    optimizers = {
        "sgd": sgd(config.learning_rate, config.weight_decay),
        "momentum": Momentum(config.learning_rate, args.momentum, config.weight_decay),
        "nag": NAG(config.learning_rate, args.momentum, config.weight_decay),
        "rmsprop": RMSprop(config.learning_rate, args.beta, args.epsilon, config.weight_decay),
        "adam": Adam(config.learning_rate, args.beta1, args.beta2, args.epsilon, config.weight_decay),
        "nadam": Nadam(config.learning_rate, args.beta1, args.beta2, args.epsilon, config.weight_decay)
    }
    return optimizers.get(name, sgd(config.learning_rate))
