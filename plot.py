from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    max_iters = 60000
    params = np.array([])
    del_theta = np.array([])
    i = 0
    #while True:
    while i < max_iters:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        params = np.append(params, np.linalg.norm(theta))
        del_theta = np.append(del_theta, np.linalg.norm(prev_theta - theta))
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    x = np.arange(0,len(params),1)
    '''
    plt.figure(figsize=(8,8))
    plt.scatter(x,params, c='blue', s=8)
    plt.title("Magnitude of Theta")
    plt.ylabel("2-Norm Theta")
    plt.xlabel("Iteration")
    #plt.show()
    plt.figure(figsize=(8,8))
    plt.scatter(x,np.log(del_theta), c='red', s=8)
    plt.title("Log-Scaled Difference")
    plt.ylabel("Log Norm Relative Iterate Decrease")
    plt.xlabel("Iteration")
    plt.show()
    '''
    return theta

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('data_a.txt')
    # Adding gaussian noise to training data/labels leads to convergence. 
    #Xa += np.random.random_sample(Xa.shape)
    #Ya += np.random.random_sample(Ya.shape)
    cmap = {-1 : 'blue', 1 : 'red'}
    plt.figure()
    plt.scatter(Xa[:,1][Ya==1], Xa[:,2][Ya==1], c = cmap[1], label = 'y=1', s=12)
    plt.scatter(Xa[:,1][Ya==-1], Xa[:,2][Ya==-1], c = cmap[-1], label = 'y=-1', s=12)
    plt.legend(loc='upper right')
    
    x = np.linspace(0,1.2,1000)
    y = -(-11.26 + 12.159*x) / 10.395
    plt.plot(x,y,'b')
    plt.title('SVM for Dataset A')
    plt.show()

    return

if __name__ == '__main__':
    main()
