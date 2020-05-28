# Developing an AI application to recognise flower species

In this project, I will develop a deep learning image classifier to recognize different species of flowers - and deploy it in a command line application. This could be used for example, in something like a phone app that tells you the name of the flower your camera is looking at.

The project has two parts:

1. Develop image classifier (in jupyter notebook image_classifier.ipynb)
2. Deploy image classifier (in python command line applications, included in this git repo)

In part 1 where we will develop our classifier, we will use the following steps:

- Load and preprocess the image dataset
- Train the image classifier on the dataset
- Use the trained classifier to predict image content

In part 2, I will then convert the work done in part 1 into two python command line applications:

- **train.py** - Will build, train, test & save a classifier that can predict flower species
- **predict.py** - Given an image of a flower and a trained classifier, will predict the species of flower

## Results

### [Project 1 - Jan 2019](https://github.com/pranath/flower_recognition/blob/master/image_classifier.ipynb)

This project used the Pytorch library, and achieved a test accuracy of 73%.

### [Project 2 - November 2019](https://github.com/pranath/flower_recognition/blob/master/image-classifier-v2.ipynb)

This project used the FastAi library, and achieved a test accuracy of 93%.

### [Project 3 - May 2020](https://github.com/pranath/flower_recognition/blob/master/image-classifier-v3.ipynb)

This project used to best model architecture from Project 2 and used it to explore how convolutions work. In particular I explore how basic convolutions work, as well as looking at how final layer convolutions can be examined (using their activations) to gain a better understanding of what the model considers important in the flower image for making its predictions.
