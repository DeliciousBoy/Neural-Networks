#install package
import matplotlib.pyplot as plt
import numpy as np 
from neural_networks.layer_dense import Layer_Dense

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')

    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)* 4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

X, y = create_data(100, 3)      
# plt.scatter(X[:,0], X[:,1] ) 
# plt.show()

# plt.scatter(X[:, 0], X[:,1], c=y, cmap='brg')
# plt.show()

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)


