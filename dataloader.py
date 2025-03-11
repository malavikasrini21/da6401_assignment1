import numpy as np
from keras.datasets import fashion_mnist, mnist

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y.astype(int)]  # Ensures integer indexing for safety

def load_data(dataset_name):
    """
    Loads and preprocesses the dataset.
    :param dataset_name: "mnist" or "fashion_mnist"
    :return: (X_train, y_train, X_test, y_test) - normalized & one-hot encoded
    """
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset name! Choose 'mnist' or 'fashion_mnist'.")

    # Convert data type to float32 for stability before normalization
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    # Flatten images and normalize
    X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0

    # One-hot encode labels
    y_train, y_test = one_hot_encode(y_train), one_hot_encode(y_test)

    return X_train, y_train, X_test, y_test
