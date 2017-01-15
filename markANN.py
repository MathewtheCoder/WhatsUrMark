import numpy as np


class Neural_Network(object):
    def __init__(self):
    #Define Hyper Parameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

    #Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)		
	
#Forward Propagation
    def forward(self, X):
        #Propagate inputs through network
        #Synapses of ANN multiplies its weight with the input and output the result.
        #So to get the activity of second layer we multiply the input matrix with weight matrix
        #X(1)*W(2) = Z(2)
        #Z(2) is a 3x3 matrix
        self.z2 = np.dot(X, self.W1)
        #Apply activation function to the input in second layer
        #A(2) = 1/1-e^-z(2)
        self.a2 = self.sigmoid(self.z2)
        #A(2)*W(2) = Z(2)
        self.z3 = np.dot(self.a2, self.W2)
        #We apply the activation function on Z(3)
        yHat = self.sigmoid(self.z3) 
        #We get a 3x1 matrix as output
        return yHat
     
    def sigmoid(self, z):
    #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

if __name__ == '__main__':

    # X = (hours sleeping, hours studying), y = Score on test
    X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)
    X = X/np.amax(X, axis=0)
    y = y/100 #Max test score is 100
    #Create an instance of neural network 
    mANN = Neural_Network()

    #Run the forward propagation
    yn = mANN.forward(X)

    #Print the output of neural network
    print("Output of ANN:",yn)
	
    #Print the original network
    print("Original data:",y)
	
