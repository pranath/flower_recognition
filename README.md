## Developing an AI application to recognise flower species

In this project, I will develop a deep learning image classifier to recognize different species of flowers - and deploy it in a command line application. This could be used for example, in something like a phone app that tells you the name of the flower your camera is looking at.

The project has two parts:

1. Develop image classifier (in this jupyter notebook)
2. Deploy image classifier (in python command line applications, included in this git repo)

In part 1 where we will develop our classifier, we will use the following steps:

- Load and preprocess the image dataset
- Train the image classifier on the dataset
- Use the trained classifier to predict image content

In part 2, I will then convert the work done in part 1 into two python command line applications:

- **train.py** - Will build, train, test & save a classifier that can predict flower species
- **predict.py** - Given an image of a flower and a trained classifier, will predict the species of flower
