# test script for the Stochastic Gradient Method (SGM)
# with diminishing stepsize
#
# here, we use a quadratic of the form f(x) = (sum_i[(x-a_i)^2])/n
# where the a_i`s are points in the linspace of [-1,1] and n the number of points
#
# INPUT:
# x0: initial point/guess
# it: the number of iterations 
#
# OUTPUT:
# x: the final point after all iterations
# fn: the function value at x

import numpy as np
import matplotlib.pyplot as plt

class Quadratic_SGM_Test:
    
    def __init__(self, x0, it):
        self.x0 = x0        
        self.it = it        

    # the SGM
    def sgm(self, coefficients_array, theoretical_minima):

        x = self.x0
        f = x**2 + theoretical_minima
        err_array = []
        err_array.append(abs(f - theoretical_minima))        
        
        for i in range(0, self.it):

            # select a random coefficient a_j, and therefore a 
            # random function fj(x) = (x - a_j)^2 as an estimator of f
            random_coefficient = np.random.choice(coefficients_array)

            # computing gradient of fj
            gradient = 2*(x - random_coefficient)

            # the diminishing stepsize
            lr = 1/(2+i)

            # the iterate update            
            x = x - lr*gradient            

            # computing function value f and the distance between f and global minima           
            f = x**2 + theoretical_minima
            err_array.append(abs(f - theoretical_minima))            

        return x, f, err_array     
    
    def quadratics_test(self):
        # setting the array of a_i's, with length 1k
        coefficients_array = np.linspace(-1,1,1000)

        # computing the theoretical global minima of our function
        theoretical_minima = np.sum(np.power(coefficients_array, 2))/1000                              

        x, f, err_array = self.sgm(coefficients_array, theoretical_minima)        

        return x, f, err_array, theoretical_minima
    
    def plot_error(self):

        x, f, err_array, theoretical_minima = self.quadratics_test()

        it_array = np.arange(0, self.it+1)        

        print('Last point x: ' + str(x))
        print('Function value at x: ' + str(f))
        print('Theoretical global minima: ' + str(theoretical_minima))

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)      
        
        ax1.set_title('MGE')
        ax1.set_ylabel('Erro')                
        ax1.plot(it_array, err_array)

        ax2.set_title('MGE- Zoomed in')         
        ax2.axis([0,len(it_array), 0, 0.2])                     
        ax2.plot(it_array, err_array)        
        
        for ax in fig.get_axes():
            ax.grid()            

        plt.show()

    def plot_expected_value(self):

        x, f, err_array, theoretical_minima = self.quadratics_test()             
        
        mean_error_array = []

        for k in range(0, self.it):
        
            kth_mean_error = np.mean(err_array[0:(k+1)])
            mean_error_array.append(kth_mean_error)        
        
        it_array = np.arange(0, self.it)           

        fig, ax1 = plt.subplots()

        ax1.grid()
        ax1.set_title('MGE')
        ax1.set_xlabel('Iterações')
        ax1.set_ylabel('Valor Esperado')        
        ax1.plot(it_array, mean_error_array, color='teal')

        plt.show()