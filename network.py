import numpy as np
from activation import get_activation
from loss import get_loss, get_loss_derivative

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation, weight_init, loss_fn, optimizer, num_layers):
        """
        Initializes a feedforward neural network with:
        - An input layer
        - Multiple hidden layers
        - An output layer with Softmax activation
        """
        self.layers = [input_size] + [hidden_size] * num_layers + [output_size]
        self.activation_fn = get_activation(activation)  # Hidden layer activation
        self.loss_fn = get_loss(loss_fn)  # Loss function
        self.loss_derivative = get_loss_derivative(loss_fn)
        self.optimizer = optimizer  # Optimizer instance

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        #debugging
        self.running_means = []
        self.running_vars = []

        self.gammas = [np.ones((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        self.betas = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]


        for i in range(len(self.layers) - 1):
            if weight_init == "random":
                W = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01
            elif weight_init == "Xavier":
                W = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(1.0 / self.layers[i])
            else:
                raise ValueError("Invalid weight initialization method!")

            self.weights.append(W)
            self.biases.append(np.zeros((1, self.layers[i + 1])))

            #debugging
            self.running_means.append(np.zeros((1, self.layers[i + 1])))
            self.running_vars.append(np.ones((1, self.layers[i + 1])))
    # def batch_norm_forward(self, Z, layer_idx):
    #     """Apply batch normalization"""
    #     epsilon = 1e-5
    #     mean = np.mean(Z, axis=0, keepdims=True)
    #     var = np.var(Z, axis=0, keepdims=True)
        
    #     # Running averages (for stability)
    #     self.running_means[layer_idx] = 0.9 * self.running_means[layer_idx] + 0.1 * mean
    #     self.running_vars[layer_idx] = 0.9 * self.running_vars[layer_idx] + 0.1 * var
        
    #     Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    #     return Z_norm

    def batch_norm_forward(self, Z, layer_idx):
        epsilon = 1e-5
        mean = np.mean(Z, axis=0, keepdims=True)
        var = np.var(Z, axis=0, keepdims=True)

        # Running averages (for stability)
        self.running_means[layer_idx] = 0.9 * self.running_means[layer_idx] + 0.1 * mean
        self.running_vars[layer_idx] = 0.9 * self.running_vars[layer_idx] + 0.1 * var

        Z_norm = (Z - mean) / np.sqrt(var + epsilon)
        return self.gammas[layer_idx] * Z_norm + self.betas[layer_idx]  # Scale and shift

    def forward(self, X):
        """Performs forward propagation through all layers."""
        self.a = [X]  # Store activations
        self.z = []   # Store linear transformations

        for i in range(len(self.weights) - 1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            if i != 0:
                z = self.batch_norm_forward(z, i)
            a = self.activation_fn(z)  # Hidden layer activation
            self.z.append(z)
            self.a.append(a)
            print(f"Layer {i+1}: Mean Activation = {np.mean(a):.4f}, Variance = {np.var(a):.4f}")

        # Output layer (Softmax)
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        a = get_activation("softmax")(z)  # Softmax for classification
        self.z.append(z)
        self.a.append(a)

        return a  # Final predictions

    # def backward(self, y_true):
    #     """Performs backpropagation through all layers."""
    #     m = y_true.shape[0]
    #     dZ = self.loss_derivative(self.a[-1], y_true)  # Output layer error

    #     dWs = []
    #     dbs = []

    #     for i in range(len(self.weights) - 1, 0, -1):
    #         dW = np.dot(self.a[i].T, dZ) / m
    #         db = np.sum(dZ, axis=0, keepdims=True) / m
    #         dWs.insert(0, dW)
    #         dbs.insert(0, db)

    #         dA = np.dot(dZ, self.weights[i].T)
    #         dZ = self.activation_fn.diff(dA)
            
    #         print(f"Layer {i}: Grad W = {np.linalg.norm(dW):.6f}, Grad b = {np.linalg.norm(db):.6f}")

    #     dW0 = np.dot(self.a[0].T, dZ) / m
    #     db0 = np.sum(dZ, axis=0, keepdims=True) / m
    #     dWs.insert(0, dW0)
    #     dbs.insert(0, db0)

    #     print(f"Input Layer: Grad W = {np.linalg.norm(dW0):.6f}, Grad b = {np.linalg.norm(db0):.6f}")
    #     # Apply gradient clipping
    #     clip_value = 5.0
    #     dWs = [np.clip(dW, -clip_value, clip_value) for dW in dWs]
    #     dbs = [np.clip(db, -clip_value, clip_value) for db in dbs]

    #     # Apply L2 regularization debug
    #     lambda_reg = 0.01  # L2 Regularization strength
    #     for i in range(len(self.weights)):
    #         dWs[i] += lambda_reg * self.weights[i]

    #     # Update weights using optimizer
    #     for i in range(len(self.weights)):
    #         self.optimizer.update(self.weights[i], self.biases[i], dWs[i], dbs[i])

    def backward(self, y_true):
        m = y_true.shape[0]
        dZ = self.loss_derivative(self.a[-1], y_true)  # Output layer error

        dWs = []
        dbs = []
        dGammas = []
        dBetas = []

        for i in range(len(self.weights) - 1, 0, -1):
            dW = np.dot(self.a[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dWs.insert(0, dW)
            dbs.insert(0, db)

            dA = np.dot(dZ, self.weights[i].T)
            dZ = self.activation_fn.diff(dA)

            # Gradients for batch normalization
            if i != 0:
                dGamma = np.sum(dZ * self.a[i], axis=0, keepdims=True) / m
                dBeta = np.sum(dZ, axis=0, keepdims=True) / m
                dGammas.insert(0, dGamma)
                dBetas.insert(0, dBeta)

            print(f"Layer {i}: Grad W = {np.linalg.norm(dW):.6f}, Grad b = {np.linalg.norm(db):.6f}")

        dW0 = np.dot(self.a[0].T, dZ) / m
        db0 = np.sum(dZ, axis=0, keepdims=True) / m
        dWs.insert(0, dW0)
        dbs.insert(0, db0)

        print(f"Input Layer: Grad W = {np.linalg.norm(dW0):.6f}, Grad b = {np.linalg.norm(db0):.6f}")

        # Apply gradient clipping
        clip_value = 5.0
        dWs = [np.clip(dW, -clip_value, clip_value) for dW in dWs]
        dbs = [np.clip(db, -clip_value, clip_value) for db in dbs]
        dGammas = [np.clip(dGamma, -clip_value, clip_value) for dGamma in dGammas]
        dBetas = [np.clip(dBeta, -clip_value, clip_value) for dBeta in dBetas]

        # Apply L2 regularization
        lambda_reg = 0.01  # L2 Regularization strength
        for i in range(len(self.weights)):
            dWs[i] += lambda_reg * self.weights[i]

        # Update weights, biases, gammas, and betas using optimizer
        for i in range(len(self.weights)):
            self.optimizer.update(self.weights[i], self.biases[i], dWs[i], dbs[i])
            if i != 0:
                self.optimizer.update(self.gammas[i - 1], self.betas[i - 1], dGammas[i - 1], dBetas[i - 1])

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """Trains the network and computes train & validation loss/accuracy."""
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch, y_batch = X_train[i:i + batch_size], y_train[i:i + batch_size]
                self.forward(X_batch)
                self.backward(y_batch)

            # Compute train & validation loss/accuracy
            train_loss, train_acc = self.evaluate(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
                  f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

    def evaluate(self, X, y):
        """Computes loss & accuracy for given dataset."""
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) * 100
        return loss, accuracy

    def predict(self, X):
        """Predicts class labels for input X."""
        return np.argmax(self.forward(X), axis=1)
