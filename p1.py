import numpy as np 

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Activation_ReLU:
    def forward(self, inputs: list[list[int]]) -> None:
        self.output = np.maximum(0, inputs)

        
# layer1 = Layer_Dense(4, 5)
# layer2 = Layer_Dense(5, 2)

# layer1.forward(X)
# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)


# input = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = [i for i in input if max(0, i)]
# print(output)