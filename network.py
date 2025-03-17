import wandb
import numpy as np
from activation import get_activation
from loss import get_loss, get_loss_derivative

class NeuralNetwork:
    def __init__(self, use_wandb,wandb_project,wandb_entity,input_size, hidden_size, output_size, activation, weight_init, loss_fn, optimizer, num_layers, lr,momentum,beta,beta1,beta2,epsilon,weight_decay):
        """
        Initializes a feedforward neural network with:
        - An input layer
        - Multiple hidden layers
        - An output layer with Softmax activation
        """
        self.use_wandb=use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.layers = [input_size] + [hidden_size] * num_layers + [output_size]
        #debug
        # print("Layers Configuration:", self.layers)

        self.activation_fn = get_activation(activation)  # Hidden layer activation
        self.loss_fn = get_loss(loss_fn)  # Loss function
        self.loss_derivative = get_loss_derivative(loss_fn)
        self.optimizer = optimizer  # Optimizer instance
        self.lr=lr

        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_init = weight_init
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        #debugging
        print(f"weight_init: {self.weight_init}")
        print(f"activation: {activation}")
        print(f"loss_fn: {loss_fn}")
        print(f"optimizer: {optimizer}")
        print(f"lr: {lr}")
        print(f"momentum: {momentum}")
        print(f"beta: {beta}")
        print(f"beta1: {beta1}")
        print(f"beta2: {beta2}")
        print(f"epsilon: {epsilon}")
        print(f"weight_decay: {weight_decay}")
        print(f"input_size: {input_size}")
        print(f"hidden_size: {hidden_size}")
        print(f"output_size: {output_size}")
        print(f"num_layers: {num_layers}")

        for i in range(len(self.layers) - 1):
            if self.weight_init == "random":
                W = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01
            elif self.weight_init == "Xavier":
                W = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(1.0 / self.layers[i])
            else:
                raise ValueError("Invalid weight initialization method!")

            self.weights.append(W)
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def forward(self, X):
        """Performs forward propagation through all layers."""
        self.a = [X]  # Store activations
        self.z = []   # Store linear transformations

        for i in range(len(self.weights) - 1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a = self.activation_fn(z)  # Hidden layer activation
            self.z.append(z)
            self.a.append(a)
            # print(f"Layer {i+1}: Mean Activation = {np.mean(a):.4f}, Variance = {np.var(a):.4f}")

        # Output layer (Softmax)
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        a = get_activation("softmax")(z)  # Softmax for classification
        self.z.append(z)
        self.a.append(a)

        return a  # Final predictions


    def backward(self, y_true):
        m = y_true.shape[0]
        dZ = self.loss_derivative(self.a[-1], y_true)  # Output layer error

        dWs = []
        dbs = []

        for i in range(len(self.weights) - 1, 0, -1):
            dW = np.dot(self.a[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dWs.insert(0, dW)
            dbs.insert(0, db)

            dA = np.dot(dZ, self.weights[i].T)
            dZ = self.activation_fn.diff(dA)

        dW0 = np.dot(self.a[0].T, dZ) / m
        db0 = np.sum(dZ, axis=0, keepdims=True) / m
        dWs.insert(0, dW0)
        dbs.insert(0, db0)

        #print(f"Input Layer: Grad W = {np.linalg.norm(dW0):.6f}, Grad b = {np.linalg.norm(db0):.6f}")

        # Apply gradient clipping
        clip_value = 5.0
        dWs = [np.clip(dW, -clip_value, clip_value) for dW in dWs]
        dbs = [np.clip(db, -clip_value, clip_value) for db in dbs]

        # Apply L2 regularization
        lambda_reg = self.weight_decay  # L2 Regularization strength
        for i in range(len(self.weights)):
            dWs[i] += lambda_reg * self.weights[i]

        # Update weights, biases using optimizer
        
        for i in range(len(self.weights)):
            #debug
            #print(f"Updating Layer {i}: Weight Shape = {self.weights[i].shape}, dW Shape = {dWs[i].shape}")
            
            if self.optimizer.__class__.__name__.lower() in ["momentum","nag","adam","rmsprop","nadam"]:
                self.optimizer.update(self.weights[i], self.biases[i], dWs[i], dbs[i],i)
            else:
                self.optimizer.update(self.weights[i], self.biases[i], dWs[i], dbs[i])    
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """Trains the network and computes train & validation loss/accuracy."""
        if self.use_wandb == "true":
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

        self.optimizer.learning_rate = self.lr
        self.optimizer.momentum = self.momentum
        self.optimizer.beta = self.beta
        self.optimizer.beta1 = self.beta1
        self.optimizer.beta2 = self.beta2
        self.optimizer.epsilon = self.epsilon
        self.optimizer.weight_decay = self.weight_decay

        ns = X_train.shape[0]
        nb = int(np.ceil(ns / batch_size))

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0

            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for b in range(nb):
                si = b * batch_size
                ei = min((b + 1) * batch_size, ns)
                X_batch = X_train[si:ei]
                y_batch = y_train[si:ei]

                # Forward pass
                activations = self.forward(X_batch)

                # Loss calculation
                y_pred = activations
                loss = self.loss_fn(y_pred, y_batch)
                accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

                # Backward pass
                self.backward(y_batch)

                epoch_loss += loss * (ei - si)
                epoch_acc += accuracy

            epoch_loss /= ns
            tr_acc = epoch_acc / ns

            # Validation
            val_loss, val_acc = self.evaluate(X_val, y_val)

            print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, Train Acc = {tr_acc:.2f}%, "
                f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

            if self.use_wandb == "true":
                wandb.log({
                    'epoch': epoch,
                    'avg_train_loss': epoch_loss,
                    'avg_valid_loss': val_loss,
                    'avg_train_acc': tr_acc,
                    'avg_valid_acc': val_acc
                })

        if self.use_wandb == "true":
            wandb.finish()

    def evaluate(self, X, y):
        """Computes loss & accuracy for given dataset."""
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

    def predict(self, X):
        """Predicts class labels for input X."""
        return np.argmax(self.forward(X), axis=1)



