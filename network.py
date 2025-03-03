import numpy as np
from activation import get_activation
from loss import get_loss, get_loss_derivative

#, optimizer -- use in class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation, weight_init, loss_fn,optimizer):
        """
        Initializes the neural network with one input layer, one hidden layer, and one output layer.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of output classes (10 for MNIST/Fashion-MNIST).
        :param activation: Activation function for the hidden layer.
        :param weight_init: Weight initialization method ("random" or "Xavier").
        :param loss_fn: Loss function ("cross_entropy" or "mean_squared_error").
        :param optimizer: Optimizer object for updating weights.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = get_activation(activation)  # Hidden layer activation
        self.loss_fn = get_loss(loss_fn)
        self.loss_derivative = get_loss_derivative(loss_fn)
        self.optimizer = optimizer  # Optimizer object

        # Weight Initialization
        if weight_init == "random":
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        elif weight_init == "Xavier":
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        else:
            raise ValueError("Invalid weight initialization method!")

        # Bias Initialization
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Performs forward propagation."""
        self.X = X  # Store input
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear transformation (input → hidden)
        self.A1 = self.activation(self.Z1)  # Activation function (hidden layer)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear transformation (hidden → output)
        self.A2 = get_activation("softmax")(self.Z2)  # Softmax output

        return self.A2  # Return predictions

    def backward(self, y_true):
        """Performs backpropagation and updates weights using optimizer."""
        m = y_true.shape[0]  # Number of samples

        # Compute gradients for output layer
        dZ2 = self.loss_derivative(self.A2, y_true)  # Error at output
        print(f"dz2:{dZ2}")
        dW2 = np.dot(self.A1.T, dZ2) / m
        print(f"dW2:{dW2}")
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        print(f"db2:{db2}")

        # Compute gradients for hidden layer
        dA1 = np.dot(dZ2, self.W2.T)  # Propagate error to hidden layer
        print(f"dA1:{dA1}")
        dZ1 = self.activation.diff(dA1)  # Activation function derivative
        print(f"dZ1:{dZ1}")
        dW1 = np.dot(self.X.T, dZ1) / m
        print(f"dW1:{dW1}")
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        print(f"db1:{db1}")

        # Print gradient magnitudes for debugging
        print(f"Grad W1: {np.linalg.norm(dW1):.6f}, Grad b1: {np.linalg.norm(db1):.6f}")
        print(f"Grad W2: {np.linalg.norm(dW2):.6f}, Grad b2: {np.linalg.norm(db2):.6f}")

        # Update weights using optimizer
        self.optimizer.update(self.W1, self.b1, dW1, db1)
        self.optimizer.update(self.W2, self.b2, dW2, db2)


    # def backward(self, y_true):
    #     """Performs backpropagation and updates weights using optimizer."""
    #     m = y_true.shape[0]  # Number of samples

    #     # Compute gradients for output layer
    #     dZ2 = self.loss_derivative(self.A2, y_true)  # Error at output
    #     dW2 = np.dot(self.A1.T, dZ2) / m
    #     db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    #     # Compute gradients for hidden layer
    #     dA1 = np.dot(dZ2, self.W2.T)  # Propagate error to hidden layer
    #     dZ1 = self.activation.diff(dA1)  # Activation function derivative
    #     dW1 = np.dot(self.X.T, dZ1) / m
    #     db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    #     # Update weights using optimizer
    #     # self.optimizer.update(self.W1, self.b1, dW1, db1)
    #     # self.optimizer.update(self.W2, self.b2, dW2, db2)

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        """Trains the neural network using mini-batch gradient descent."""
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)  # Shuffle data
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                self.forward(X_batch)
                self.backward(y_batch)

            # Compute loss & accuracy after each epoch
            y_pred = self.forward(X_test)
            loss_value = self.loss_fn(y_pred, y_test)
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) * 100

            print(f"Epoch {epoch+1}: Loss = {loss_value:.4f}, Accuracy = {accuracy:.2f}%")

    def predict(self, X):
        """Predicts class labels for input X."""
        return np.argmax(self.forward(X), axis=1)
