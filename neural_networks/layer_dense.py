import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        """
        Initialize the layer with random weights and biases

        Parameters:
            n_inputs  : Number of input features.
            n_neurons : Number of neurons in the layer.
        """
        # Initialize weights with a standard normal distribution scaled by 0.10
        self.weight = 0.10 * np.random.randn(n_inputs, n_neurons) 
        # Initialize biases with zeros
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> None:
        """
        Perform a forward pass through the layer.

        Parameters:
            inputs  : Input data.
        """
        self.output = np.dot(inputs, self.weight) + self.biases