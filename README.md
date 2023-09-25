# Gated Recurrent Unit (GRU) Text Autocomplete 
GRU implementation for text autocomplete and prediction using a Gated Recurrent Unit neural network model. ( last updated : May 2023)

## Overview
This project implements a Gated Recurrent Unit (GRU) neural network model for the task of text autocomplete and prediction, based on the seminal 2014 paper by Cho et al. that originally proposed the GRU architecture. The goal is to predict the next words in a text corpus by learning patterns from training data. 

## Background
Traditional recurrent neural networks (RNNs) suffer from the vanishing gradient problem which makes it difficult for them to capture long-term dependencies in sequential data like text. The GRU was designed to address this by incorporating gating mechanisms that better regulate the flow of information. Cho et al.'s research showed that GRUs outperform standard RNNs on tasks requiring memory of previous inputs like language modeling and translation.

## Implementation
The core GRU model is implemented in Python using more primitive Pytorch and Numpy functions for efficient numerical computations. It consists of an input layer, GRU hidden layers with update and reset gates, and an output layer with a softmax activation to produce a probability distribution over possible next words. The model is trained on a text corpus.

## Usage
Train the model on a text corpus of your choice. This can be changed on lines 47 and 48 of the .py file "GRU_based_autocomplete.py"

This will train the GRU for 30 epochs on the origin_of_species extract from works on Charles Darwin and save the trained model weights.

The model can then be used to generate text continuations by sampling from the predicted next-word distributions: 


## Results 
Preliminary results validating the original GRU paper show the model is able to capture long-term dependencies and learn patterns to successfully generate multi-word text continuations, demonstrating its effectiveness for text modeling tasks.

## Future Work
Areas for improvement include developing beam search to produce higher quality generations, and testing on larger modern corpora.

