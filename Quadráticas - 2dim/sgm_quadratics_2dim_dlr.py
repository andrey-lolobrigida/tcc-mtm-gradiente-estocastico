# test script for the Stochastic Gradient Method (SGM)
# with diminishing stepsize
#
# here, we use a quadratic of the form f = (sum_i[(x-a_i)^2])/n + (sum_j[(y-b_i)^2])/n
# where the a_i`s and b_i`s are points in the linspace of [-1,1] and n the number of points
#
# INPUT:
# x0: initial point/guess (2 dimensional numpy array)
# it: the number of iterations 
#
# OUTPUT:
# p: the final point after all iterations
# fn: the function value at x

import numpy as np
import matplotlib.pyplot as plt

class Quadratic_SGM_Test:
    
    def __init__(self, p0, it):
        self.p0 = p0        
        self.it = it        

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

            # the diminishing stepsize
            lr = 1/(2+i)

            # the iterate update            
            p = p - lr*gradient            

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

        p, x_array, y_array, f, err_array, theoretical_minima = self.quadratics_test()
        it_array = np.arange(0, self.it+1)        

        print('Last point p: ' + str(p))
        print('Function value at x: ' + str(f))
        print('Theoretical global minima: ' + str(theoretical_minima))

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)      
        
        ax1.set_title('SGM')
        ax1.set_ylabel('Error')                
        ax1.plot(it_array, err_array)

        ax2.set_title('SGM - Zoomed in')         
        ax2.axis([0,len(it_array), 0, 0.2])                     
        ax2.plot(it_array, err_array)        
        
        for ax in fig.get_axes():
            ax.grid()            

        plt.show()

    def plot_contours(self):

        p, x_array, y_array,f, err_array, theoretical_minima = self.quadratics_test()
        it_array = np.arange(0, self.it+1)

        xlist = np.linspace(-1.5, 1.5, self.it)                 
        X, Y = np.meshgrid(xlist, xlist)

        Z = X**2 + Y**2 + theoretical_minima

        fig, ax = plt.subplots(1,1)
        cp = ax.contourf(X, Y, Z, 20)
        fig.colorbar(cp)
        ax.plot(x_array, y_array)

        plt.show()

