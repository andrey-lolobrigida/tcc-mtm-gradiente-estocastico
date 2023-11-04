#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar 

class NeuralNetwork:

    def __init__(self, layers_list, epochs, learning_rate, training_set):
        self.layers_list= layers_list
        self.epochs = epochs
        self.learning_rate = learning_rate        
        self.training_set = training_set

    def compute_activation(self, affine_vector):
        # sigmoid                                
        activation_vector = 1/(1 + np.exp(-affine_vector))    

        return activation_vector
    
    def compute_loss(self, activation_vector, output_vector):
        # using mean squared error
        loss_vector = (activation_vector - output_vector)**2 

        loss_eval = np.sum(loss_vector)

        return loss_vector, loss_eval
    
    def forward_pass(self, input_vector, output_vector, param_array):

        forward_pass_array = []
        activation_vector = input_vector
        forward_pass_array.append(activation_vector)       

        for l in range(0, len(param_array)):
            
            affine_vector = (param_array[l][0].dot(activation_vector) + param_array[l][1])            
            activation_vector = self.compute_activation(affine_vector)            

            forward_pass_array.append(activation_vector)            

        loss_vector, loss_eval = self.compute_loss(activation_vector, output_vector)        

        forward_pass_array.append(loss_vector)
        forward_pass_array.append(loss_eval)

        return forward_pass_array
    
    def backpropagation(self, param_array, forward_pass_array, output_vector):        

        backpropagation_array = []        
        
        layer_dim = len(forward_pass_array[-3])
        derivatives_vector = (forward_pass_array[-3]*(np.ones(layer_dim) - forward_pass_array[-3]))  
        s_vector = 2*(forward_pass_array[-3] - output_vector)*derivatives_vector                     

        backpropagation_array.insert(0, s_vector)               

        for l in range(-1,-len(param_array),-1):
            
            layer_dim = len(forward_pass_array[l-3])                                
            delta_vector = np.transpose(param_array[l][0]).dot(backpropagation_array[l])
            derivatives_vector = forward_pass_array[l-3]*(np.ones(layer_dim) - forward_pass_array[l-3])
            delta_vector = delta_vector*derivatives_vector                     
            
            backpropagation_array.insert(0, delta_vector)            

        return backpropagation_array

    def parameters_update(self, param_array, forward_pass_array, backpropagation_array):        

        param_index = 0
        for param in param_array:            
            
            matrix = param[0]
            bias = param[1]            

            for row in range(0, matrix.shape[0]):
                bias[row] = bias[row] - self.learning_rate*backpropagation_array[param_index][row]

                for column in range(0, matrix.shape[1]):
                    row_element = backpropagation_array[param_index][row]
                    column_element = forward_pass_array[param_index][column]                    

                    matrix[row, column] = matrix[row, column] - self.learning_rate*row_element*column_element                    
                          
            param_index = param_index + 1

        return param_array
    
    def training(self):
        
        it_number = self.epochs*len(self.training_set[0])
        param_array = []

        for l in range(1, len(self.layers_list)):                        
            
            layer_matrix = np.random.uniform(-1, 1, (self.layers_list[l], self.layers_list[l-1]))
            layer_biases = np.zeros(self.layers_list[l])

            param_array.append([layer_matrix, layer_biases])        
        
        loss_array = []

        print('Begin training.')        

        with alive_bar(it_number) as bar:            

            for epoch in range(0, self.epochs):                
                
                sample_array = np.arange(len(self.training_set[0]))                
                np.random.shuffle(sample_array)
                
                # the SGM
                for sample in sample_array:                                    

                    input_vector = self.training_set[0][sample].flatten()/255          
                    output_vector = np.zeros(10)
                    output_vector[self.training_set[1][sample]] = 1                              

                    forward_pass_array = self.forward_pass(input_vector, output_vector, param_array)
                    backpropagation_array = self.backpropagation(param_array, forward_pass_array, output_vector)

                    param_array = self.parameters_update(param_array, forward_pass_array, backpropagation_array)
                    loss_array.append(forward_pass_array[-1])                    

                    bar()

        print('Training complete!')
        
        return param_array, loss_array

    def plot_error(self):       

        param_array, loss_array = self.training()

        it_number = self.epochs*len(self.training_set[0])
        it_array = np.arange(0, it_number)        
        mean_error_array = np.zeros(it_number)        

        print('Computing mean error array.')

        with alive_bar(it_number-1) as bar:

            mean_error_array[0] = loss_array[0]                             

            for k in range(1, it_number):                

                mean_error_array[k] = (k*mean_error_array[k-1] + loss_array[k])/(k+1)                

                bar()
        
        print('Plotting loss.')        
        
        loss_plot = plt.plot(it_array, mean_error_array)        

        return param_array, loss_plot   

        












    

     