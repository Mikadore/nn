from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

def empty_list(N):
    return [None for _ in range(N)]

@dataclass
class ForwardPropActivations:
    input: np.ndarray
    """ Unprocessed input """

    Z: list[np.ndarray]
    """ Linear activations """

    A: list[np.ndarray]
    """ Actual activations """

    @staticmethod
    def empty(input) -> "ForwardPropActivations":
        return ForwardPropActivations(np.copy(input), [], [])

@dataclass
class Gradients:
    dW: list[np.ndarray]
    """ Deltas for the layer weights """

    dB: list[np.ndarray]
    """ Deltas for the layer biases """

    @staticmethod
    def empty(N):
        return Gradients(empty_list(N), empty_list(N))
    
@dataclass
class Layer:
    W: np.ndarray
    """ The weights for this layer """
    B: np.ndarray
    """ The biases for this layer """
    N: int
    """ Number of neurons in the layer """
    
class Hyperparams(ABC):
    @staticmethod
    @abstractmethod
    def loss(activations, targets):
        raise NotImplemented
    
    @staticmethod
    @abstractmethod
    def output_activation(input):
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def output_derivative(activations, linear_activations, targets):
        raise NotImplemented
    
    @staticmethod
    @abstractmethod
    def activation(layer):
        raise NotImplemented
    
    @staticmethod
    @abstractmethod
    def activation_derivative(layer):
        raise NotImplemented
    
class NeuralNet:
    hyperparams: Hyperparams
    """ Hyperparameters for this network """

    layers: list[Layer]
    """ The hidden layers of this network """
    
    N: int
    """ The number of hidden layers """

    input_size: int
    """ Input layer size """
    
    output_size: int
    """ Output layer size """

    def __init__(self, hyperparams: Hyperparams, input_size: int, layer_sizes: list[int], output_size: int):
        assert len(layer_sizes) != 0, "There must be at least 1 hidden layer"
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.output_size = output_size
        self.N = len(layer_sizes)
        
        self.layers = list()
        
        # Xavier initialization,
        # each layer is scaled by the 
        # inverse of the previous layer's size
        prev_layer_size = input_size
        for size in [*layer_sizes, output_size]:
            limit = np.sqrt(6 / (prev_layer_size + size))
            W = np.random.uniform(-limit, limit, (prev_layer_size, size))
            B = np.zeros((1, size))
            N = size
            prev_layer_size = size
            self.layers.append(Layer(W, B, N))

    def copy(self) -> "NeuralNet":
        nn = NeuralNet(self.hyperparams, self.input_size, [l.N for l in self.layers[:-1]], self.output_size)
        nn.layers = [Layer(np.copy(l.W), np.copy(l.B), l.N) for l in self.layers]
        return nn
    
    def forward(self, input) -> ForwardPropActivations:
        # Make input a Matrix if it isn't.
        # This is required so that we can multiply
        # it with other matrices
        if input.ndim == 1:
            input = input[np.newaxis, :]

        activations = ForwardPropActivations.empty(input)

        # Process hidden layers
        prev_layer = input
        for i in range(self.N):
            Z = np.dot(prev_layer, self.layers[i].W) + self.layers[i].B
            #A = sigmoid(Z)
            A = self.hyperparams.activation(Z)
            prev_layer = A
            activations.Z.append(Z) 
            activations.A.append(A)

        # Process the output layer
        Z_out = np.dot(prev_layer, self.layers[self.N].W) + self.layers[self.N].B
        #A_out = softmax(Z_out)
        A_out = self.hyperparams.output_activation(Z_out)
        activations.Z.append(Z_out)
        activations.A.append(A_out) 

        return activations
    
    def backward(self, activations: ForwardPropActivations, expected_output: np.ndarray) -> Gradients:
        gradients = Gradients.empty(self.N + 1)

        dZ = self.hyperparams.output_derivative(activations.A[-1], activations.Z[-1], expected_output)
     
        for i in reversed(range(self.N + 1)):
            A_prev = activations.A[i - 1] if i != 0 else activations.input

            dW = np.dot(A_prev.T, dZ)
            dB = np.sum(dZ, axis=0, keepdims=True)
            
            if i > 0:
                dA_prev = np.dot(dZ, self.layers[i].W.T)
                dZ = dA_prev * self.hyperparams.activation_derivative(activations.Z[i - 1])
    
            gradients.dW[i] = dW
            gradients.dB[i] = dB

        return gradients

    def train(self, inputs, results, learning_rate) -> float:
        assert len(inputs) == len(results)
        activations = self.forward(inputs)
        gradient = self.backward(activations, results)
        for i in range(self.N + 1):
            self.layers[i].W -= learning_rate * gradient.dW[i]
            self.layers[i].B -= learning_rate * gradient.dB[i]
        return self.hyperparams.loss(activations.A[-1], results)

def sigmoid(n):
    return 1 / (1 + np.exp(-n))

def sigmoid_derivative(n):
    s = sigmoid(n) 
    return s * (1 - s)

def softmax(n):
    e_n = np.exp(n - np.max(n, axis=1, keepdims=True))
    return e_n / e_n.sum(axis=1, keepdims=True)

def categorical_cross_entropy(predictions, targets):
    eps = 1e-15  # To prevent log(0)
    predictions = np.clip(predictions, eps, 1 - eps)
    N = predictions.shape[0]
    return -np.sum(targets * np.log(predictions))/N

class Sigmoid_CCE_Softmax:
    @staticmethod
    def loss(activations, targets):
        return categorical_cross_entropy(activations, targets)
    
    @staticmethod
    def output_activation(input):
        return softmax(input)

    @staticmethod
    def output_derivative(activations, linear_activations, targets):
        return activations - targets
    
    @staticmethod
    def activation(layer):
        return sigmoid(layer)
    
    @staticmethod
    def activation_derivative(layer):
        return sigmoid_derivative(layer)
    
Hyperparams.register(Sigmoid_CCE_Softmax)

def relu(n):
    return np.maximum(0, n)

def relu_derivative(n):
    return (n > 0).astype(float)

def mean_squared_error(activations, targets):
    return np.mean((targets - activations) ** 2)

def mean_squared_error_derivative(activations, targets):
    return (2 / len(activations) * (activations - targets))

class Relu_MSE_Softmax:
    @staticmethod
    def loss(activations, targets):
        return mean_squared_error(activations, targets)
    
    @staticmethod
    def output_activation(input):
        return softmax(input)

    @staticmethod
    def output_derivative(activations, linear_activations, targets):
        dA = mean_squared_error_derivative(activations, targets)
        return dA * relu_derivative(linear_activations)
    
    @staticmethod
    def activation(layer):
        return sigmoid(layer)
    
    @staticmethod
    def activation_derivative(layer):
        return sigmoid_derivative(layer)