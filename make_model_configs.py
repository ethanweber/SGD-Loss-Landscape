import argparse
import pprint
from sklearn.datasets import make_regression
from os import path
import numpy as np

parser = argparse.ArgumentParser(description="")
parser.add_argument('--num-layers', type=int, default=None, help="Number of layers to add in the model.", required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--dataset-path", type=str, help="Path to dataset.", required=True)
parser.add_argument("--config-path", type=str, help="Path to store JSON config.", default="configs/")
parser.add_argument("--config-name", type=str, help="Name of the config file.", required=True) 
parser.add_argument("--model-path", type=str, help="Path to store model weights.", default="models/")
parser.add_argument("--epochs", type=int, help="Number of epochs to train for.", default=1)
parser.add_argument("--num-gpus", type=int, help="Number of GPUs to use", default=1) 

def main(args):
    print("Generating dataset with args:")
    pprint.pprint(args)

    # make folder if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    training_data = np.load(args.dataset_path)
    feature_dim = training_data.shape[1]

    bs = args.batch_size
    if args.batch_size=="-1":
        bs = training_data.shape[0]

    layers = [torch.nn.Linear(feature_dim, feature_dim*2), torch.nn.LeakyReLU()]
    layers.append([item for _ in range(args.num_layers) for item in [torch.nn.Linear(feature_dim, feature_dim*2), torch.nn.LeakyReLU()]]) 
    layers.append(torch.nn.Linear(100, 1))

    model = torch.nn.Sequential(*layers)
    model_file = path.join(args.model_path, args.config_name)
    torch.save(model, model_file)
 
    model_definition = {
        "model_path": model_file 
        "dataset": args.dataset_path,
        "output_folder": args.model_path,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": bs, 
        "epochs": args.epochs,
        "num_gpus": args.num_gpus
    }

    output_path = path.join(args.config_path, args.config_name)
    with open(output_path, "wb") as f:
        json.dumps(f, model_definition)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
