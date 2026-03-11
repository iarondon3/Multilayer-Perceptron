import numpy as np
import activation_functions


def initialize_parameters(input_neurons, output_neurons, hidden_layers, neurons_per_layer):
    layers = [input_neurons] + ([neurons_per_layer] * hidden_layers) + [output_neurons]
    
    weights = []
    biases = []
    
    for i in range(len(layers) - 1):
        
        limit = np.sqrt(2.0 / layers[i])
        weight_matrix = np.random.randn(layers[i], layers[i+1]) * limit
        weights.append(weight_matrix)
        
        bias_matrix = np.zeros((1, layers[i+1]))  
        biases.append(bias_matrix)
    
    return input_neurons, output_neurons, hidden_layers, neurons_per_layer, weights, biases


def forward_propagation(dataset, weights, biases):

    activations = [dataset]
        
    layer_output = dataset

    for w, b in zip(weights, biases):

        z = np.dot(layer_output, w) + b
            
        layer_output = activation_functions.sigmoid_function(z)

        activations.append(layer_output)
        
    return layer_output, activations


def backpropagation(weights, biases, activations, target, learning_rate):
    
    current_output = activations[-1]
    output_error = target - current_output
    output_delta = output_error * activation_functions.sigmoid_derivative(current_output)

    deltas = [output_delta]

    # Backwards
    for i in range(len(weights) - 1, 0, -1):
        
        next_w = weights[i]
        next_delta = deltas[-1]
        
        current_activation = activations[i]
        
        hidden_layer_error = next_delta.dot(next_w.T)
        
        hidden_delta = hidden_layer_error * activation_functions.sigmoid_derivative(current_activation)
        
        deltas.append(hidden_delta)

    # Learning
    deltas.reverse()

    for i in range(len(weights)):
        w = weights[i]
        b = biases[i]
        delta = deltas[i]
        activation = activations[i]

        # Gradient calculation
        weight_change = np.dot(activation.T, delta)
        bias_change = np.sum(delta, axis=0, keepdims=True)

        # Apply learning
        weights[i] = w + (learning_rate * weight_change)
        biases[i] = b + (learning_rate * bias_change)

    return weights, biases


def train_model(inputs, targets, weights, biases, learning_rate, epochs, start_counter=0):

    print(f"Starting training for {epochs} epochs (Cumulative total at the end: {start_counter + epochs})...")

    errors = []

    for epoch in range(epochs):

        prediction, history = forward_propagation(inputs, weights, biases)

        weights, biases = backpropagation(weights, biases, history, targets, learning_rate)

        current_error = np.mean((targets - prediction) ** 2)
        errors.append(current_error)


        if epoch % 1000 == 0 or epoch == (epochs - 1):
            print(f"Epoch {epoch}: Error = {current_error:.6f}")

    print("Training finished")
    return weights, biases, errors
