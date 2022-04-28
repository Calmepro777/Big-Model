# Experiment Intro

In this experiment, CLIP is used as the teacher network to train a resnet-18 model on CIFAR-100 dataset

`train.py`: Model traning.

`Model_eval.ipynb`: Visualize model predictions, evaluate prediction accuracy of the model.

## Steps
 - Create text input based on the class names in the traning dataset (e.g. apple ==> This is a photo of apple)
 - Encode text input with the pre-trained text encoder of CLIP model
 - Encode batched images with the pre-trained image encoder of CLIP model (ResNet 50)
 - Calculate the cosine similarities between image features and text features
 - Use the cosine similarities as the output of the teacher network
 - The distilled knowledge is then used to train the student network by minimizing the KLDivergence between the teacher network output and student network    output
