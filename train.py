import wandb
import argparse
import numpy as np
from keras.datasets import fashion_mnist

import argparse
import wandb
import numpy as np
from dataloader import load_data
from network import NeuralNetwork
from optimizer import get_optimizer

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a feedforward neural network")
    
    # WandB Args
    parser.add_argument("--use_wandb", type=str, default="false")
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname")

    # Dataset
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist")

    # # Training Parameters
    parser.add_argument("-n", "--saz", type=int, default=10,help="sample images")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=48)

    # # Loss & Optimizer
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    
    # # Optimizer Parameters
    # parser.add_argument("-m", "--momentum", type=float, default=0.5)
    # parser.add_argument("-beta", "--beta", type=float, default=0.5)
    # parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    # parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    # parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)

    # # Model Parameters
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, default=128)
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")

    return parser.parse_args()

# --- Main Execution ---
def main():
    args = parse_args()
    
    # Initialize Weights & Biases
    if args.use_wandb.lower() == "true":
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Load Data
    X_train, y_train, X_test, y_test = load_data(args.dataset)
 
    #small test split
    X_smp_test = X_test[:args.saz]
    y_smp_test = y_test[:args.saz]

    splits=int(0.85 * X_train.shape[0])
    # train split
    # X_smp_train = X_train[(args.saz//3):args.saz]
    # y_smp_train = y_train[(args.saz//3):args.saz]
    X_smp_train = X_train[:splits]
    y_smp_train = y_train[:splits]
    # valid split 
    # X_smp_valid = X_smp_train[:args.saz//3]
    # y_smp_valid = y_smp_train[:args.saz//3]
    X_smp_valid = X_train[splits:]
    y_smp_valid = y_train[splits:]

    # print(X_smp_train.shape)
    # print(X_smp_valid.shape)
    # print(X_smp_test.shape)
    # print(y_smp_train.shape)
    # print(y_smp_valid.shape)
    # print(y_smp_test.shape)

    print(f"Train set shape: {X_smp_train.shape}, {y_smp_train.shape}")
    print(f"Validation set shape: {X_smp_valid.shape}, {y_smp_valid.shape}")
    print(f"Test set shape: {X_smp_test.shape}, {y_smp_test.shape}")
    
    optimizer = get_optimizer(args.optimizer, None, args)

    # Initialize Model
    model = NeuralNetwork(
        input_size=X_smp_train.shape[1],
        # hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        loss_fn=args.loss,
        optimizer=optimizer,
        num_layers=args.num_layers
    )
    
    # # Get Optimizer
    # get_optimizer(args.optimizer, model, args)

    # # Train Model
    # model.train(X_smp_train, y_smp_train, X_smp_test, y_smp_test, epochs=args.epochs, batch_size=args.batch_size, loss=args.loss, optimizer=optimizer)
    #model.train(X_smp_train, y_smp_train, X_smp_test, y_smp_test, epochs=args.epochs, batch_size=args.batch_size)
    model.train(X_smp_train, y_smp_train,X_smp_valid, y_smp_valid, epochs=args.epochs, batch_size=args.batch_size)
    # # Finish WandB
    # wandb.finish()

if __name__ == "__main__":
    main()
