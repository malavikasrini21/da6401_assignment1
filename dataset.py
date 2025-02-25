import wandb
import numpy as np
from keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="da6401_assignment1", name="fashion_mnist_dataset")

# Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for labels
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Convert images to a format compatible with wandb (add color channel)
train_images = np.expand_dims(train_images, axis=-1)  # Shape (28, 28) → (28, 28, 1)

# Log sample images to W&B
wandb.log({"examples": [wandb.Image(train_images[i], caption=class_names[train_labels[i]]) for i in range(10)]})

# Finish logging
wandb.finish()
