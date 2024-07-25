# Simple Neural Network Implementation (No libraries)

This Python script demonstrates a basic implementation of a single-layer neural network. The purpose of this code is to showcase the fundamental concepts of neural networks, including forward propagation, backpropagation, and gradient descent.

## Features

- Implementation of a simple neural network with two input neurons and one output neuron
- Training process using gradient descent
- Prediction functionality for new inputs
- Customizable hyperparameters (learning rate, number of epochs)

## How It Works

1. **Initialization**: The network starts with random weights and a bias term.

2. **Forward Propagation**: The `predict` function calculates the output of the neural network for given inputs.

3. **Training**:
   - The network is trained for a specified number of epochs.
   - In each epoch, it makes predictions for all training inputs.
   - The cost (mean squared error) is calculated to measure the network's performance.

4. **Backpropagation**:
   - The script calculates the gradients of the cost with respect to the weights and bias.
   - It updates the weights and bias using the calculated gradients and the learning rate.

5. **Testing**: After training, the network can make predictions on new, unseen data.

## Customization

You can easily adapt this code for different datasets and problems by modifying:
- The input data and target values
- The number of input features
- The learning rate and number of epochs

## Educational Purpose

This implementation is designed for educational purposes to help understand the basics of neural networks. It's not optimized for performance or intended for production use. For real-world applications, consider using established machine learning libraries like TensorFlow or PyTorch.

## Getting Started

1. Ensure you have Python installed on your system.
2. Copy the code into a Python file (e.g., `neural_network.py`).
3. Run the script using a Python interpreter.

Feel free to experiment with the code and adapt it to your learning needs!
