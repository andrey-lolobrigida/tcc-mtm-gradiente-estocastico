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


import math
import numpy as np
import matplotlib.pyplot as plt

class And_Operator_Test:

    def __init__(self, theta0, it, lr):
        self.theta0 = theta0        
        self.it = it
        self.lr = lr

    def assemble_training_set(self):

        x_set = np.array([[1.,1.],
                          [-1.,1.],
                          [1.,-1.],
                          [-1.,-1.]])
        
        y_set = np.array([1.,-1.,-1.,-1.])       

        return x_set, y_set

    def training(self):

        x_set, y_set = self.assemble_training_set()

        w1 = self.theta0[0]
        w2 = self.theta0[1]
        b = -1

        sample_indexes = np.arange(0,4)

        w1_array = np.zeros(self.it)
        w2_array = np.zeros(self.it)        

        loss_array = np.zeros(self.it)
        full_loss_array = np.zeros(self.it)
        gradient_norm_array = np.zeros(self.it)       

        for j in range(0, self.it):

            random_sample_index = np.random.choice(sample_indexes)

            w1_array[j] = w1
            w2_array[j] = w2                          

            # forward pass 

            z = (np.matmul(np.array([w1, w2]), x_set[random_sample_index,:]) + b)    
            a = math.tanh(2*z)                       

            loss_eval = (a - y_set[random_sample_index])**2

            # backpropagation                                  

            s = 2*(a - y_set[random_sample_index])*(2/(np.cosh(2*z)**2))     # versão de MSE

            # compute full loss

            z_array = np.zeros(4)
            a_array = np.zeros(4)            

            for i in range(0,4):
                z_array[i] = (np.matmul(np.array([w1, w2]), x_set[i,:]) + b)
                a_array[i] = math.tanh(2*z_array[i])
            
            component_loss_array = a_array - y_set
            full_loss_eval = np.sum(np.power(component_loss_array,2))/4

            full_loss_array[j] = full_loss_eval

            # compute gradient at sample and its norm

            g1 = (s*x_set[random_sample_index,0])
            g2 = (s*x_set[random_sample_index,1])
            gradient_norm = np.linalg.norm([g1, g2])

            gradient_norm_array[j] = gradient_norm
            
            # theta parameters update

            w1 = w1 - self.lr*g1
            w2 = w2 - self.lr*g2                    

            loss_array[j] = loss_eval

        return w1_array, w2_array, loss_array, full_loss_array, gradient_norm_array    

    def plot_error(self):

        w1_array, w2_array, loss_array, full_loss_array, gradient_norm_array = self.training()

        it_array = np.arange(0, self.it)


        print('Valor de w1: ' + str(w1_array[-1]), 'Valor de w2: ' + str(w2_array[-1]))
        print('Erro na amostra: ' + str(loss_array[-1]), 'Erro total: ' + str(full_loss_array[-1]))

        fig, ax1 = plt.subplots()

        ax1.grid()        
        ax1.plot(it_array, full_loss_array)
        
        plt.show()

    def plot_mgn(self):
        # 'mgn' stands for mean gradient norm

        w1_array, w2_array, loss_array, full_loss_array, gradient_norm_array = self.training()

        mgn_array = []

        for k in range(0, self.it):

            kth_mgn = np.mean(gradient_norm_array[0:(k+1)])
            mgn_array.append(kth_mgn)

        it_array = np.arange(0, self.it)

        fig, ax1 = plt.subplots()

        ax1.grid()        
        ax1.plot(it_array, mgn_array)

        plt.show()

    def plot_contours(self):

        w1_array, w2_array, loss_array, full_loss_array, gradient_norm_array = self.training()

        xlist = np.linspace(-0.5, 2.5, self.it)                 
        X, Y = np.meshgrid(xlist, xlist)

        Z = ((np.tanh(X + Y - 1) - 1)**2 + (np.tanh(X - Y - 1) + 1)**2 + (np.tanh(-X + Y - 1) + 1)**2 + (np.tanh(-X - Y - 1) + 1)**2)
        Z = Z/4        

        fig, ax = plt.subplots(1,1)
        cp = ax.contourf(X, Y, Z, 20)
        fig.colorbar(cp)
        ax.plot(w1_array, w2_array)

        print('Valor de w1: ' + str(w1_array[-1]), 'Valor de w2: ' + str(w2_array[-1]))
        print('Erro na amostra: ' + str(loss_array[-1]), 'Erro total: ' + str(full_loss_array[-1]))

        plt.show()