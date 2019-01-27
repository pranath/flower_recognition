import argparse
from model import FlowerPredictionModel
from utils import load_data

# Set command line argument parser
parser = argparse.ArgumentParser(
    description='Train model to predict flower image names',
)
# Required command line arguments
parser.add_argument('data_directory', help="Directory to read images from", action="store")
# Optional command line arguments
parser.add_argument("--save_dir", help="Directory to save trained model", action="store")
parser.add_argument("--arch", help="Architecture of model", action="store")
parser.add_argument("--learning_rate", help="Learning rate", action="store", type=float)
parser.add_argument("--hidden_units", help="Number of hidden units for classifier", action="store", type=int)
parser.add_argument('--epochs', help="Number of epochs to train model", action="store", type=int)
parser.add_argument('--gpu', help="Use GPU if available", action='store_true')

args = parser.parse_args()

# Load data
trainloader, validloader, testloader, class_to_idx = load_data(args.data_directory)

# Create model object
model = FlowerPredictionModel(args.gpu)

# Define model architecture parameters
if args.arch is not None:
    arch = args.arch
else:
    arch="vgg16"
if args.learning_rate is not None:
    learning_rate = args.learning_rate
else:
    learning_rate=0.003
if args.hidden_units is not None:
    hidden_units = args.hidden_units
else:
    hidden_units=15000
    
# Build new model
model.build(arch, learning_rate, hidden_units)

# Define model training parameters
if args.epochs is not None:
    epochs = args.epochs
else:
    epochs=4
    
# Train model
model.train(epochs, trainloader, validloader, class_to_idx)

# Test model
model.test(testloader)

# Define model save parameters
if args.save_dir is not None:
    save_dir = args.save_dir
else:
    save_dir=""
    
# Save model
model.save(save_dir)
