import argparse
import wandb
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from dataloader import *
from network import NeuralNetwork
from optimizer import get_optimizer
from activation import get_activation

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a feedforward neural network")
    
    # WandB Args
    parser.add_argument("--use_wandb", type=str, default="false")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_assignment1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="hyperparameter_optimization")

    # Dataset
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist")

    # Training Parameters
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)

    # # Loss & Optimizer
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    
    # Optimizer Parameters
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-beta", "--beta", type=float, default=0.9)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.005)

    # # Model Parameters
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="Xavier")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, default=256)
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh")

    return parser.parse_args()

def wandb_sweep():
    with wandb.init() as run:
        config = wandb.config
        run_name = f"ac_{config.activation}_hl_{config.num_layers}_hs_{config.hidden_size}_bs_{config.batch_size}_op_{config.optimizer}_ep_{config.epochs}"
        wandb.run.name = run_name

        optimizer = get_optimizer(config.optimizer, None, config,args)
        model = NeuralNetwork(
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            input_size=X_smp_train.shape[1],
            hidden_size=config.hidden_size,
            output_size=10,
            activation=config.activation,
            weight_init=config.weight_init,
            loss_fn=config.loss,
            optimizer=optimizer,
            num_layers=config.num_layers,
            lr=config.learning_rate,
            momentum=args.momentum,
            beta=args.beta,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            weight_decay=config.weight_decay,
        )
        
        model.train(X_smp_train, y_smp_train, X_smp_valid, y_smp_valid, X_test, y_test, target_classes,epochs=config.epochs, batch_size=config.batch_size)
        
        # # Test Model
        # test_predictions = model.predict(X_test)
        # test_predictions_oh = one_hot_encode(test_predictions)
        # test_predictions_oh = get_activation("softmax")(test_predictions_oh)
        # test_loss = model.loss_fn(test_predictions_oh, y_test)
        # test_accuracy = np.mean(np.argmax(test_predictions_oh, axis=1) == np.argmax(y_test, axis=1)) * 100

        # # Log test metrics to WandB
        # if args.use_wandb == "true":
        #     wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        #     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        #     wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

        #     # Log Confusion Matrix
        #     y_true = np.argmax(y_test, axis=1)  # Convert y_test to class indices
        #     y_pred = np.argmax(test_predictions_oh, axis=1)  # Convert predictions to class indices
        #     wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        #                 probs=None,  # No probabilities, just class indices
        #                 y_true=y_true,  # 1D array of true class indices
        #                 preds=y_pred,  # 1D array of predicted class indices
        #                 class_names=target_classes  # List of class names
        #     )})
        
        # if args.use_wandb == "true":
        #     wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        #     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        #     wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

        #     # Log Confusion Matrix
        #     y_pred = np.argmax(test_predictions_oh, axis=1)
        #     wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_test, preds=y_pred, class_names=target_classes)})
        #     wandb.finish()
# --- Main Execution ---
def main(args: argparse.Namespace):
    if args.use_wandb == "true":
        wandb.login()
        sweep_config = {
            'method': 'bayes',
            'name' : 'sweep cross entropy',
            'metric': {
                'name': 'avg_valid_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'epochs': {
                    'values': [5,10,15,20]
                },
                'num_layers': {
                    'values': [2,3,4]
                },
                'learning_rate': {
                    'values': [0.001, 0.0001]
                },'hidden_size':{
                    'values': [32,64,128,256]
                },
                'weight_decay': {
                    'values': [0,0.0005,0.005,0.5]
                },'batch_size':{
                    'values': [16,32,64,128,256]
                },'optimizer':{
                    'values': ['sgd','momentum','nag','rmsprop','adam','nadam']
                },'weight_init': {
                    'values': ['random','Xavier']
                },'activation':{
                    'values': ['sigmoid','tanh','ReLu']
                },'loss':{
                    'values':['cross_entropy']
                }
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)

    if args.use_wandb == "true":
        wandb.agent(sweep_id, function=wandb_sweep, count=50)
        wandb.finish()
    else:    
        optimizer = get_optimizer(args.optimizer, None, None, args)

        # Initialize Model
        model = NeuralNetwork(
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            input_size=X_smp_train.shape[1],
            hidden_size=args.hidden_size,
            output_size=10,
            activation=args.activation,
            weight_init=args.weight_init,
            loss_fn=args.loss,
            optimizer=optimizer,
            num_layers=args.num_layers,
            lr=args.learning_rate,
            momentum=args.momentum,
            beta=args.beta,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            weight_decay=args.weight_decay,
        )
        
        model.train(X_smp_train, y_smp_train,X_smp_valid, y_smp_valid, X_test,y_test,target_classes,epochs=args.epochs, batch_size=args.batch_size)
    
        # Test Model
        test_predictions = model.predict(X_test)
        test_predictions=one_hot_encode(test_predictions)
        test_predictions = get_activation("softmax")(test_predictions)  # Ensure predictions are probabilities
        test_loss = model.loss_fn(test_predictions, y_test)
        test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1)) * 100

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == 'fashion_mnist':
        target_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    elif args.dataset == 'mnist':
        target_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # Load Data
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    splits=int(0.90 * X_train.shape[0])
    # train split
    X_smp_train = X_train[:splits]
    y_smp_train = y_train[:splits]
    # valid split 
    X_smp_valid = X_train[splits:]
    y_smp_valid = y_train[splits:]

    print(f"Train set shape: {X_smp_train.shape}, {y_smp_train.shape}")
    print(f"Validation set shape: {X_smp_valid.shape}, {y_smp_valid.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")

    main(args)
