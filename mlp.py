import numpy as np
import random
from alive_progress import alive_it as bar
from .ActivationFunctions import *
import Optimizers

class mlp:
    def __init__(self, Neurons, AF_Type, Weights=None, Biases=None):
        self.Neurons = Neurons
        self.LastLayer = len(Neurons) - 1
        self.AF_Type = AF_Type
        self.Weights = Weights
        self.Biases = Biases
        self.LearningRate = 0.0001
        self.cost_function = lambda Acti, Output: (Acti - Output)*2
        self.Optimizer = Optimizers.Momentum
        self.info = True
        
        #Global variables handled by the class (Don't mess with them)
        self.Biases_mat = None
        self.OptimizerCache = [0 for _ in range(self.LastLayer*2)]

    def Bias_convert(self, matrix_width): #Converts single bias vector to a bias matrix
        Biases_mat = [None]
        for i in range(1, len(self.Biases)):
            Biases_mat.append(np.matmul(self.Biases[i], np.full((1, matrix_width), 1)))
        return Biases_mat
        
    def Build_matrices(self):
        self.Weights = [None for i in range(self.LastLayer+1)]
        self.Biases = [None for i in range(self.LastLayer+1)]
        for layer in range(1, self.LastLayer+1):
            if self.AF_Type[layer] == "Sigmoid" or self.AF_Type[layer] == "tanh":
                self.Xavior_init(layer)
            if self.AF_Type[layer] == "ReLU" or self.AF_Type[layer] == "LeakyReLU":
                self.He_init(layer)

    def Xavior_init(self, layer): #For sigmoid and tanh
        self.Weights[layer] = np.random.normal(size=(self.Neurons[layer], self.Neurons[layer-1]))*np.sqrt(1/self.Neurons[layer])
        self.Biases[layer] = np.zeros((self.Neurons[layer], 1))

    def He_init(self, layer): #For ReLU and Leaky ReLU
        self.Weights[layer] = np.random.normal(size=(self.Neurons[layer], self.Neurons[layer-1]))*np.sqrt(2/self.Neurons[layer])
        self.Biases[layer] = np.full((self.Neurons[layer], 1), 0.1)
        
    def FeedForward(self, init_activations, use_precalculated_bias=False): #Computes all the z vectors as well as activations
        z_vectors = [None]
        activations = [init_activations]
        if use_precalculated_bias == True:
            biases = self.Biases_mat
        else:
            biases = self.Bias_convert(init_activations.shape[1])
        #Calculating the next layer of activations and z vector
        for layer in range(1, self.LastLayer+1):
            z_vectors.append(np.matmul(self.Weights[layer], activations[layer-1]) + biases[layer])
            activations.append(AF(z_vectors[layer], self.AF_Type[layer]))
        return activations, z_vectors
        
    def Backpropagate(self, activations, z_vectors, desired_output, training_batch, pass_on_gradient=False):
        ErrorVector = [AF_dv(z_vectors[self.LastLayer], self.AF_Type[self.LastLayer])*self.cost_function(activations[self.LastLayer], desired_output)]
        for j in range(1, self.LastLayer):
            i = self.LastLayer - j
            ErrorVector.insert(0, np.matmul(self.Weights[i+1].T, ErrorVector[0])*AF_dv(z_vectors[i], self.AF_Type[i]))
        ErrorVector.insert(0, None)
            
        #Update matrices
        for i in range(1, self.LastLayer+1):
            weight_gradient = np.matmul(ErrorVector[i], activations[i-1].T) / training_batch
            Grad, self.OptimizerCache = self.Optimizer(self.LearningRate, weight_gradient, self.OptimizerCache[i-1])
            self.Weights[i] = self.Weights[i] - Grad

        for i in range(1, self.LastLayer+1):
            bias_gradient = np.matmul(ErrorVector[i], np.full((training_batch, training_batch), 1)) / training_batch
            Grad, self.OptimizerCache = self.Optimizer(self.LearningRate, bias_gradient, self.OptimizerCache[i-1 + self.LastLayer])
            self.Biases_mat[i] = self.Biases_mat[i] - Grad
        
        if pass_on_gradient == True:
            return np.matmul(self.Weights[1].T, ErrorVector[1])
    
    def train(self, TrainingData, TrainingTimes=1000, TrainingBatch=1000):
        data_length = len(TrainingData[0])-1
        self.Biases_mat = self.Bias_convert(TrainingBatch)

        for Iterations in bar(range(TrainingTimes)):
            #Randomize
            rand = [random.randint(0, data_length) for i in range(TrainingBatch)]
            activations = []
            desired_output = []
            for index in rand:
                activations.append(TrainingData[0][index])
                desired_output.append(TrainingData[1][index])
            image_activations = np.array(activations).T
            desired_output = np.array(desired_output).T
            
            activations, z_matrix = self.FeedForward(image_activations, True)
            self.Backpropagate(activations, z_matrix, desired_output, TrainingBatch)

            if self.info == True:
                #Calculating accuracy of batch
                wrong = 0
                outputs = (step(activations[self.LastLayer]) - desired_output).T
                for i in range(TrainingBatch):
                    wrong += abs(np.linalg.norm(outputs[i].T))
                print((wrong*100)/TrainingBatch)

        for layer in range(1, self.LastLayer+1):
            self.Biases[layer] = np.array([self.Biases_mat[layer][:, 0]]).T