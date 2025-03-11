import numpy as np

# Mean Squared Error (MSE) Loss
def mean_squared_error(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

# MSE Derivative
def mean_squared_error_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

# Cross-Entropy Loss
def cross_entropy(y_pred, y_true):
    eps = 1e-8  # To prevent log(0)
    return -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]

# Cross-Entropy Derivative
def cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true

# Get Loss Function
def get_loss(loss_name):
    if loss_name == "mean_squared_error":
        return mean_squared_error
    elif loss_name == "cross_entropy":
        return cross_entropy
    else:
        raise ValueError("Invalid loss function name!")

# Get Loss Derivative
def get_loss_derivative(loss_name):
    if loss_name == "mean_squared_error":
        return mean_squared_error_derivative
    elif loss_name == "cross_entropy":
        return cross_entropy_derivative
    else:
        raise ValueError("Invalid loss function name!")
