# Imports here
import torch
import numpy as np
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import json

# Utility functions for images

# Load images for training
def load_data(data_dir):
    ''' Load images required for training & testing deep learning model.
    '''

    # Data split into training, validation & testing sets
    # Define image folders
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Return data loaders
    return trainloader, validloader, testloader, train_data.class_to_idx

# Load map for converting class indicies to flower names
def load_label_map():
    ''' Load map for converting class indicies to flower names
    '''

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

        return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model

    # Load image from file & get dimensions
    im = Image.open(image)
    width, height = im.size

    # Scale image
    # (determine which is the shortest side to restrict scaling)
    if width > height:
        # Height is shortest to be set to max of 256
        im.thumbnail((width, 256), Image.ANTIALIAS)
    else:
        # Width is shortest to be set to max of 256
        im.thumbnail((256, height), Image.ANTIALIAS)

    # Crop center 224x224
    new_width, new_height = im.size
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    im = im.crop((left, top, right, bottom))

    # Convert colour channel values from 0-255 to 0-1
    np_image = np.array(im) / 255

    # Normalise colour channels
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std_dev

    # Re-order dimensions of numpy array
    np_image = np_image.transpose((2, 0, 1))

    return np_image
