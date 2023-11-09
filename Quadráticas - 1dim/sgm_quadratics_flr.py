# test script for the Stochastic Gradient Method (SGM)
# with fixed stepsize
#
# here, we use a quadratic of the form f = (sum_i[(x-a_i)^2])/n
# where the a_i`s are points in the linspace of [-1,1] and n the number of points
#
# INPUT:
# x0: initial point/guess
# it: the number of iterations
# lr: learning rate or stepsize 
#
# OUTPUT:
# x: the final point after all iterations
# fn: the function value at x

import numpy as np
import matplotlib.pyplot as plt

class Quadratic_SGM_Test:
    
    def __init__(self, x0, it, lr):
        self.x0 = x0        
        self.it = it
        self.lr = lr

    # the SGM
    def sgm(self, coefficients_array, theoretical_minima):

        x = self.x0
        f = x**2 + theoretical_minima
        err_array = np.zeros(self.it+1)
        err_array[0] = abs(f - theoretical_minima)
        
        for j in range(1, self.it+1):            
            # select a random coefficient a_j, and therefore a 
            # random function fj = (x - a_j)^2 as an estimator of f
            random_coefficient = np.random.choice(coefficients_array)

            # computing gradient of fj
            gradient = 2*(x - random_coefficient)

            # the iterate update
            x = x - self.lr*gradient            

            # computing function value f and the distance between f and global minima           
            f = x**2 + theoretical_minima
            err_array[j] = abs(f - theoretical_minima)

        return x, f, err_array
    
    def true_grad(self, theoretical_minima):
        # an implementation of the deterministic gradient method for comparison purposes
        w = self.x0
        g = w**2 + theoretical_minima
        err_array_grad = np.zeros(self.it+1)
        err_array_grad[0] = abs(g - theoretical_minima)

        for j in range(1, self.it+1):

            w = w - self.lr*(2*w)
            g = w**2 + theoretical_minima
            err_array_grad[j] = abs(g - theoretical_minima)

        return err_array_grad    
    
    def quadratics_test(self):
        # setting the array of a_i's, with length 1k
        coefficients_array = np.linspace(-1,1,1000)

        # computing the theoretical global minima of our function
        theoretical_minima = np.sum(np.power(coefficients_array, 2))/1000                              

        x, f, err_array = self.sgm(coefficients_array, theoretical_minima)

        error_array_grad = self.true_grad(theoretical_minima)

        return x, f, err_array, error_array_grad, theoretical_minima
    
    def plot_error(self):

        x, f, err_array, error_array_grad, theoretical_minima = self.quadratics_test()        
        
        it_array = np.arange(0, self.it+1)        

        print('Last point x: ' + str(x))
        print('Function value at x: ' + str(f))
        print('Theoretical global minima: ' + str(theoretical_minima))

        fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharex=True)

        plt.rc('axes', titlesize=22)        
        plt.rc('legend', fontsize=14)                       
        
        ax1.set_title('MGE')
        ax1.set_ylabel('Erro')
        ax1.yaxis.label.set_fontsize(18)               
        ax1.plot(it_array, err_array)               

        ax2.set_title('MGE - Zoomed in')         
        ax2.axis([0, len(it_array), 0, 0.2])                     
        ax2.plot(it_array, err_array)        
        
        ax3.set_title('MGE vs Gradiente')
        ax3.set_xlabel('Iterações')
        ax3.set_ylabel('Erro')
        ax3.xaxis.label.set_fontsize(18)
        ax3.yaxis.label.set_fontsize(18)         
        ax3.plot(it_array, err_array) 
        ax3.plot(it_array, error_array_grad)
        ax3.legend(['MGE', 'Gradiente'], loc='upper right')

        ax4.set_title('MGE vs Gradiente - Zoomed in')
        ax4.set_xlabel('Iterações')
        ax4.xaxis.label.set_fontsize(18)            
        ax4.axis([0, len(it_array), 0, 0.2])                     
        ax4.plot(it_array, err_array)
        ax4.plot(it_array, error_array_grad)
        ax4.legend(['MGE', 'Gradiente'], loc='upper right')

        for ax in fig.get_axes():
            ax.grid()            

        plt.show()

    def plot_expected_value(self):

        x, f, err_array, error_array_grad, theoretical_minima = self.quadratics_test()             
        
        mean_error_array = []

        for k in range(0, self.it):
        
            kth_mean_error = np.mean(err_array[0:(k+1)])
            mean_error_array.append(kth_mean_error)        
        
        it_array = np.arange(0, self.it)           

        fig, ax1 = plt.subplots()

        plt.rc('axes', titlesize=22)        
        plt.rc('legend', fontsize=14) 

        ax1.grid()
        ax1.set_title('MGE')
        ax1.set_xlabel('Iterações')
        ax1.set_ylabel('Média dos Erros')
        ax1.xaxis.label.set_fontsize(18)
        ax1.yaxis.label.set_fontsize(18)         
        ax1.plot(it_array, mean_error_array, color='teal')

        plt.show()