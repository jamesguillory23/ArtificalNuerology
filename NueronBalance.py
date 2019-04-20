
from numpy import exp, array, random, dot
import numpy as np

#Creation of the matrix
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
#Generates random seed
random.seed(1)
#Gives a custom weight to measure. 3 x 1 with the value range of -1 to 1 the mean is 0
synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))

#create class for the Neutral network
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    #define the self made value of the nueral networks thinking structure.
    def __init__(self, x, y):
        self.input      = x
        self.weights1    = np.random.rand(self.input.shape[1], 4)
        self.weights2    = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)


    def feedForward(self):
        self.layer1     = sigmoid(np.dot(self.input, self.weights1))
        self.output     = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        #This appliies the chained ruleset that defines the derivitive
        #of a loss function without recieving conflict backlash from weight1 or weight2
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        #Puts a threshold on how far it can take its thinking process.
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                              self.weights2.T) * sigmoid_derivative(self.layer1)))

        # updates the weights the derivitive slope and the loss functions
        self.weights1 += d_weights1
        self.weights2 += d_weights2

#This declaration is made for any main function containing a matrix equation and to in line learn what its doing.
#main progrom run. Set aside from class made above.
if __name__ == "__main__":
    #creates more matrixes for the Ai to solve
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(X, y)

    for i in range(1500):

        nn.feedForward()
        nn.backprop()

    print(nn.output)

    #Creation of a separate equation