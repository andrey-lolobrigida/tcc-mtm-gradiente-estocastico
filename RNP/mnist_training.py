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

from keras.datasets import mnist
from neural_network import NeuralNetwork
from alive_progress import alive_bar 

(train_X, train_y), (test_X, test_y) = mnist.load_data()

layers_list = [784,196,49,10]
epochs_number = 5
learning_rate = 0.1
training_set = [train_X, train_y]

neural_network = NeuralNetwork(layers_list, epochs_number, learning_rate, training_set)
param_array, loss_plot = neural_network.plot_error()

score = 0

print('Evaluating accuracy.')

with alive_bar(len(test_X)) as bar:

    for k in range(len(test_X)):        

        input_vector = test_X[k].flatten()/255
        output_vector = np.zeros(10)            
        output_vector[test_y[k]] = 1    

        forward_pass_array = neural_network.forward_pass(input_vector, output_vector, param_array)
        max_value = np.max(forward_pass_array[-3])
        index = np.where(forward_pass_array[-3] == max_value)[0][0]
        
        if index == test_y[k]:
            score += 1            

        bar()    

accuracy = score/len(test_X)

print('Accuracy is: '+ str(100*accuracy) + '%')
plt.grid()
plt.show()