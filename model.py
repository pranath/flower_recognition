# Imports here
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
import sys
import os
import json

# Define class for flower prediction model

class FlowerPredictionModel:

    # Model class constructor
    def __init__(self, gpu):
        ''' Initialise model object
        '''
        # Set device
        if (gpu):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"



    # Function to build model
    def build(self, arch, learning_rate, hidden_units):
        ''' Function to build model
        '''

        print('Building model...')
        # Select & load pre-trained model
        try:
            arch = arch.lower()
            self.model = models.__dict__[arch](pretrained=True)
            self.arch = arch
        except:
            print("Model " + arch + " not recognised: please refer to documentation for valid model names in pytorch ie vgg16 https://pytorch.org/docs/stable/torchvision/models.html")
            sys.exit()

        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        # Freeze parameters of pre-trained model part by removing gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Determine classifier input units for selected model
        if hasattr(self.model, "classifier"):
            try:
                classifier_input_neurons = self.model.classifier[0].in_features
            except TypeError:
                classifier_input_neurons = self.model.classifier.in_features
        elif hasattr(self.model, "fc"):
            classifier_input_neurons = self.model.fc.in_features
        else:
            print("Unable to determine classifier input units number - unable to create model")
            return
        # Classifier architecture parameters
        classifier_output_neurons = 102
        classifier_dropout = 0.2

        # Build new classifier for recognising flowers to work with model
        self.model.classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(classifier_input_neurons, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(classifier_dropout)),
                                        ('fc2', nn.Linear(hidden_units, classifier_output_neurons)),
                                        ('output', nn.LogSoftmax(dim=1))]))

        # Define model loss function
        self.criterion = nn.NLLLoss()

        # Define training function: only train the classifier parameters, feature parameters are frozen
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)

        # Move model to current device
        self.model.to(self.device)



    # Function to train model
    def train(self, epochs, trainloader, validloader, class_to_idx):
        ''' Function to train model
        '''

        print('Training model...')
        # Set variables
        self.epochs = epochs
        self.training_steps = 0
        training_loss = 0
        print_every = 20
        self.model.class_to_idx = class_to_idx

        # Train network
        # Ensure notebook session stays active through long runs
        with active_session():

            # For each training pass of whole dataset/epoch
            for epoch in range(epochs):

                print(f"Epoch {epoch+1}")
                print("-------")

                # For each training batch/step of images & labels
                for inputs, labels in trainloader:

                    # Increment training steps count
                    self.training_steps += 1
                    # Move data and label tensors to device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Clear gradients
                    self.optimizer.zero_grad()
                    # Do forward pass through network
                    logps = self.model(inputs)
                    # Calculate loss for whole network
                    loss = self.criterion(logps, labels)
                    # Calculate gradients for each element to be trained by network (weights & biases)
                    loss.backward()
                    # Do back-propogation step: apply negative gradients to weights & biases
                    self.optimizer.step()
                    # Accumulate training loss
                    training_loss += loss.item()
                    # Every 20 training steps, validation check & output stats
                    if self.training_steps % print_every == 0:

                        valid_loss = 0
                        accuracy = 0
                        # Switch to evaluation mode - dropout inactive
                        self.model.eval()
                        # Disable gradients - not needed for modal validation/prediction
                        with torch.no_grad():

                            # For each validation batch of images & labels
                            for inputs, labels in validloader:

                                # Move data and label tensors to device
                                inputs, labels = inputs.to(self.device), labels.to(self.device)
                                # Do forward pass through network
                                logps = self.model.forward(inputs)
                                # Calculate loss for network
                                batch_loss = self.criterion(logps, labels)
                                # Accumulate validation loss
                                valid_loss += batch_loss.item()

                                # Calculate stats
                                # Get actual probabilties output from network for this batch
                                ps = torch.exp(logps)
                                # Get top probability/prediction for each image in batch
                                top_p, top_class = ps.topk(1, dim=1)
                                # Check each prediction against label (accuracy)
                                equals = top_class == labels.view(*top_class.shape)
                                # Calculate mean accuracy for this batch
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        # Output stats for current training step
                        print(f"Training step {self.training_steps}")
                        print(f"Training loss: {training_loss/print_every:.3f} - "
                              f"Validation loss: {valid_loss/len(validloader):.3f} - "
                              f"Validation accuracy: {accuracy/len(validloader):.3f}")
                        # Validation end - reset training loss & set model back to training mode
                        training_loss = 0
                        self.model.train()

    # Function to test model
    def test(self, testloader):
        ''' Function to test model
        '''

        print('Testing model...')
        accuracy = 0
        # Switch to evaluation mode - dropout inactive
        self.model.eval()
        # Disable gradients - not needed for modal testing/prediction
        with torch.no_grad():

            # For each test batch of images & labels
            for inputs, labels in testloader:

                # Move data and label tensors to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Do forward pass through network
                logps = self.model.forward(inputs)

                # Calculate stats
                # Get actual probabilties output from network for this batch
                ps = torch.exp(logps)
                # Get top probability/prediction for each image in batch
                top_p, top_class = ps.topk(1, dim=1)
                # Check each prediction against label (accuracy)
                equals = top_class == labels.view(*top_class.shape)
                # Calculate mean accuracy for this batch
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            else:
                # Output accuracy for entire test dataset
                print(f"Test accuracy: {accuracy/len(testloader):.3f}")


    # Function to save model
    def save(self, save_dir):
        ''' Function to save model
        '''

        print('Saving model...')
        # If save dir set
        if (save_dir):
            save_dir = save_dir + '/'
            # If it does not exist
            if (not os.path.isdir(save_dir)):
                # Make dir
                try:
                    os.mkdir(save_dir)
                except OSError:
                    print ("Creation of the directory %s failed" % save_dir)
                    print ("Model was not saved")
                    sys.exit()

        # Define checkpoint parameters
        checkpoint = {'class_to_idx': self.model.class_to_idx,
                    'model_state_dict': self.model.state_dict(),
                    'arch': self.arch,
                    'learning_rate': self.learning_rate,
                    'hidden_units': self.hidden_units,
                    'epochs': self.epochs,
                    'training_steps': self.training_steps}

        # Save it
        torch.save(checkpoint, save_dir + 'checkpoint.pth')

    # Function to save model
    def load(self, save_dir):
        ''' Function to load model
        '''

        print('Loading model...')
        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(save_dir + 'checkpoint.pth')
        else:
            checkpoint = torch.load(save_dir + 'checkpoint.pth', map_location=lambda storage, loc: storage)
        # Create model
        self.build(checkpoint['arch'], checkpoint['learning_rate'], checkpoint['hidden_units'])
        # Load classifier state values from checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']

    def predict(self, np_image, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        print('Model predicting...')
        # Convert image to tensor
        image_tensor = torch.from_numpy(np_image)
        # Add batch dimension to tensor
        image_tensor = image_tensor.unsqueeze_(0)
        # Convert to float tensor
        image_tensor = image_tensor.float()

        # Switch to evaluation mode - dropout inactive
        self.model.eval()
        # Disable gradients - not needed for model prediction
        with torch.no_grad():

            # Do forward pass through network
            logps = self.model.forward(image_tensor)
            # Get actual probabilties output from network for this image
            ps = torch.exp(logps)
            # Get topk probability/prediction for this image
            top_p, top_class = ps.topk(topk, dim=1)
            top_p = top_p.numpy()
            top_class = top_class.numpy()
            # Invert class map
            idx_to_class = {j: i for i, j in self.model.class_to_idx.items()}
            # Map indexes to get true class indexes
            top_classes = [idx_to_class[index] for index in top_class[0]]

            # Return probabilties and classes
            return top_p[0], top_classes


    def predict_image(self, image_path, np_image, top_k, category_names_json):

        print('Testing model prediction...')
        # Get image file parts
        image_filename = image_path.split('/')[-2]

        # Get prediction of image
        probs, classes = self.predict(np_image, top_k)
        print(" ")
        # If category names set
        if (category_names_json):
            with open(category_names_json, 'r') as f:
                cat_to_name = json.load(f)
                classes = [cat_to_name[x] for x in classes]
                print("Actual flower category: " + cat_to_name[image_filename])

        print("Categories predicted")
        print(classes)
        print("Probabilities of categories predicted")
        print(probs)
