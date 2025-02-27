import wandb
import argparse
import numpy as np
from keras.datasets import fashion_mnist

import argparse
import wandb
import numpy as np
from dataloader import load_data
from network import NeuralNetwork
# from optimizers.py import get_optimizer

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
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=1)

    # # Loss & Optimizer
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    
    # # Optimizer Parameters
    # parser.add_argument("-m", "--momentum", type=float, default=0.5)
    # parser.add_argument("-beta", "--beta", type=float, default=0.5)
    # parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    # parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    # parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    # parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)

    # # Model Parameters
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
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
 
    #test split
    X_smp_test = X_test[:args.saz]
    y_smp_test = y_test[:args.saz]

    # train split
    X_smp_train = X_train[(args.saz//3):args.saz]
    y_smp_train = y_train[(args.saz//3):args.saz]
    # valid split 
    X_smp_valid = X_smp_train[:args.saz//3]
    y_smp_valid = y_smp_train[:args.saz//3]

    print(X_smp_train.shape)
    print(X_smp_valid.shape)
    print(X_smp_test.shape)
    print(y_smp_train.shape)
    print(y_smp_valid.shape)
    print(y_smp_test.shape)
    

    # Initialize Model
    model = NeuralNetwork(
        input_size=X_smp_train.shape[1],
        # hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        loss_fn=args.loss
    )
    
    # # Get Optimizer
    # optimizer = get_optimizer(args.optimizer, model, args)

    # # Train Model
    # model.train(X_smp_train, y_smp_train, X_smp_test, y_smp_test, epochs=args.epochs, batch_size=args.batch_size, loss=args.loss, optimizer=optimizer)
    model.train(X_smp_train, y_smp_train, X_smp_test, y_smp_test, epochs=args.epochs, batch_size=args.batch_size)

    # # Finish WandB
    # wandb.finish()

if __name__ == "__main__":
    main()
