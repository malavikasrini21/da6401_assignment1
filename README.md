# DA6401 Assignment 1

## Overview
This project implements a Feed Forward Neural Network. You can find a detailed experiment report and visualizations on the [Wandb Report](https://wandb.ai/).

## Links
- [GitHub Repository](https://github.com/malavikasrini21/da6401_assignment1)
- [Wandb Report](https://wandb.ai/malavika_da24s010-indian-institute-of-technology-madras/da6401_assignment1/reports/DA6401-Assignment-1--VmlldzoxMTgyMTQzMg)

## Code Organization

The code is organized into several modules, each serving a specific function in the model pipeline. Below is a brief overview of the project structure:

### 1. `network.py`
Contains the implementation of the neural network architecture. This file defines the layers, activations, and forward pass of the model.

### 2. `optimizer.py`
Defines the custom optimization algorithms used for training the model, including any modifications to standard optimizers like Adam or SGD.

### 3. `train.py`
Main script for training the model. It handles the training loop, including model initialization, loss calculation, backpropagation, and validation steps.

### 4. `activation.py`
Includes various activation functions like ReLU, Sigmoid, etc. used in the neural network layers.

### 5. `dataloader.py`
Handles data loading, preprocessing, and batching for training and evaluation. This module supports different datasets for model training.

### 6. `dataset.py`
Contains classes or functions to load the specific dataset(s) required for training the model.

### 7. `loss.py`
Defines the custom loss functions used in training, such as Mean Squared Error (MSE), Cross-Entropy, etc.

### 8. `README.md`
This file, which provides details about the project and how to use the code.



