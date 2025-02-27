import wandb
import numpy as np
from keras.datasets import fashion_mnist

# Initializing wandb
wandb.init(project="da6401_assignment1", name="fashion_mnist_dataset")

# Loading dataset
(x_tr_imgs, y_tr_labels), (x_tst_imgs, y_tst_labels) = fashion_mnist.load_data()

# fashion_mnist classes
target_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

x_tr_imgs = np.expand_dims(x_tr_imgs, axis=-1)  # Shape (28, 28) → (28, 28, 1)

img=[]
for i in range(len(target_classes)):
    img.append(wandb.Image(x_tr_imgs[i], caption=target_classes[y_tr_labels[i]]))

wandb.log({"Sample Images of Each class in Fashion_MNIST Dataset": img})
wandb.finish()
