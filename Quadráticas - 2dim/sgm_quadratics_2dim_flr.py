# test script for the Stochastic Gradient Method (SGM)
# with fixed stepsize
#
# here, we use a quadratic of the form f = (sum_i[(x-a_i)^2])/n + (sum_j[(y-b_i)^2])/n
# where the a_i`s and b_i`s are points in the linspace of [-1,1] and n the number of points
#
# INPUT:
# p0: initial point/guess (2 dimensional numpy array)
# it: the number of iterations
# lr: learning rate or stepsize 
#
# OUTPUT:
# p: the final point after all iterations
# fn: the function value at x

import numpy as np
import matplotlib.pyplot as plt

class Quadratic_SGM_Test:
    
    def __init__(self, p0, it, lr):
        self.p0 = p0        
        self.it = it
        self.lr = lr

    # the SGM
    def sgm(self, coefficients_array, theoretical_minima):

        p = self.p0
        x_array = np.zeros(self.it)
        y_array = np.zeros(self.it) 

        f = p[0]**2 + p[1]**2 + theoretical_minima
        err_array = []
        err_array.append(abs(f - theoretical_minima))        
        
        for i in range(0, self.it):

            # select a random coefficient a_j, and therefore a 
            # random function fij = (x - a_i)^2 + (y - a_j)^2 as an estimator of f
            a_coefficient = np.random.choice(coefficients_array)
            b_coefficient = np.random.choice(coefficients_array)

            # save point coordinates

            x_array[i] = p[0]
            y_array[i] = p[1]

            # computing gradient of fj
            gradient = np.array([2*(p[0] - a_coefficient), 2*(p[1] - b_coefficient)])            

            # the iterate update            
            p = p - self.lr*gradient            

            # computing function value f and the distance between f and global minima           
            f = p[0]**2 + p[1]**2 + theoretical_minima
            err_array.append(abs(f - theoretical_minima))            

        return p, x_array, y_array, f, err_array       
    
    def quadratics_test(self):
        # setting the array of a_i's, with length 1k
        coefficients_array = np.linspace(-1,1,1000)

        # computing the theoretical global minima of our function
        theoretical_minima = np.sum(np.power(coefficients_array, 2))/500                              

        p, x_array, y_array, f, err_array = self.sgm(coefficients_array, theoretical_minima)        

        return p, x_array, y_array, f, err_array, theoretical_minima
    
    def plot_error(self):

        x, f, err_array, theoretical_minima = self.quadratics_test()        
        
        it_array = np.arange(0, self.it+1)        

        print('Last point x: ' + str(x))
        print('Function value at x: ' + str(f))
        print('Theoretical global minima: ' + str(theoretical_minima))

        fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharex=True)        
        
        ax1.set_title('SGM')
        ax1.set_ylabel('Error')                
        ax1.plot(it_array, err_array)               

        ax2.set_title('SGM - Zoomed in')         
        ax2.axis([0, len(it_array), 0, 0.2])                     
        ax2.plot(it_array, err_array)        
        
        ax3.set_title('SGM vs Gradient')
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Error')        
        ax3.plot(it_array, err_array) 
        ax3.plot(it_array, error_array_grad)
        ax3.legend(['SGM', 'Gradient'], loc='upper right')

        ax4.set_title('SGM vs Gradient - Zoomed in')
        ax4.set_xlabel('Iterations')          
        ax4.axis([0, len(it_array), 0, 0.2])                     
        ax4.plot(it_array, err_array)
        ax4.plot(it_array, error_array_grad)
        ax4.legend(['SGM', 'Gradient'], loc='upper right')

        for ax in fig.get_axes():
            ax.grid()            

        plt.show()

    def plot_expected_value(self):

        x, f, err_array, error_array_grad, theoretical_minima = self.quadratics_test()
             
        otimality_gap = theoretical_minima*self.lr 
        mean_error_array = []

        for k in range(0, self.it):
        
            kth_mean_error = np.mean(err_array[0:(k+1)])
            mean_error_array.append(kth_mean_error)        
        
        it_array = np.arange(0, self.it)
        og_array = np.full(len(it_array), otimality_gap)       

        fig, ax1 = plt.subplots()

        ax1.grid()
        ax1.set_title('SGM')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Expected Value')                 
        ax1.plot(it_array, og_array, color='red')
        ax1.plot(it_array, mean_error_array, color='teal')

        plt.show()

    def plot_contours(self):

        p, x_array, y_array, f, err_array, theoretical_minima = self.quadratics_test()
        it_array = np.arange(0, self.it+1)

        xlist = np.linspace(-1.5, 1.5, self.it)                 
        X, Y = np.meshgrid(xlist, xlist)

        Z = X**2 + Y**2 + theoretical_minima

        fig, ax = plt.subplots(1,1)
        cp = ax.contourf(X, Y, Z, 20)
        fig.colorbar(cp)
        ax.plot(x_array, y_array)

        plt.show()