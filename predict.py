import argparse
from model import FlowerPredictionModel
from utils import process_image

# Set command line argument parser
parser = argparse.ArgumentParser(
    description='Predict flower image names',
)
# Required command line arguments
parser.add_argument('image_path', help="Path to flower image", action="store")
parser.add_argument('checkpoint', help="Directory to read model from", action="store")
# Optional command line arguments  --gpu
parser.add_argument("--category_names", help="Json file for mapping of categories to real names", action="store")
parser.add_argument('--top_k', help="Return top K most likely classes", action="store", type=int)
parser.add_argument('--gpu', help="Use GPU if available", action='store_true')

args = parser.parse_args()

# Create model object
model = FlowerPredictionModel(args.gpu)

# Load model
model.load(args.checkpoint)

# Process image
np_image = process_image(args.image_path)

# Define model predicting parameters
if args.top_k is not None:
    top_k = args.top_k
else:
    top_k = 4

if args.category_names is not None:
    category_names = args.category_names
else:
    category_names=""

# Predict given image on model
model.predict_image(args.image_path, np_image, top_k, category_names)

# Example
# python predict.py flowers/test/70/image_05308.jpg ./ --top_k 5 --category_names cat_to_name.json

