import numpy as np
import random
import matplotlib.pyplot as plt

MAX_ITER = 5
EPS = 0.01

def compute_centers(data, distances, centers, k):
    m,n = data.shape
    q = np.zeros((m,k))
    for i in range(m):
        d_min = np.amin(distances[i,:])
        for ii in range(k):
            d = (d_min / np.maximum(distances[i,ii], EPS))**3 * d_min
            if (np.isnan(d)):
                print(i)
                print(distances[i,:])
            d_ = 1
            for j in range(k):
                 d_ += (d_min / np.maximum(distances[i,j], EPS))**2
            d_ -= 1
            if (np.isnan(d / d_)):
                print(i)
                print(d)
                print(d_)
            q[i,ii] = d / (d_**2)
          
    for i in range(k):
        #print("Printing q...")
        #print(np.isnan(q[:,i])) 
        #q[:,i] /= np.sum(q[:,i],axis=0)    
        #print(np.sum(q[:,i],axis=0))
        #exit() 
        for j in range(m):
            centers[:,i] += q[j,i] * data[j,:] 
    return centers

def compute_distances(data, centers, k):
    m,n = data.shape
    D = np.zeros((m,k))
    for i in range(m):
        for j in range(k):
            D[i][j] = np.linalg.norm(data[i,:].toarray() - centers[:,j], 2)
    return D

def cluster(data, k):
    ## Randonly initialize centers.
    m,n = data.toarray().shape
    centers = np.zeros((n,k))
    for i in range(k):
        row_index = random.randint(0, (m-1))
        centers[:,i] = np.random.random_sample(n) #data[row_index, :].toarray()
    iter = 0
    while(iter < MAX_ITER):
        print(iter)
        iter += 1
        dist = compute_distances(data, centers, k) 
        centers = compute_centers(data, dist, centers, k)
    print(dist)
    print(centers)
    plt.figure()
    plt.plot(centers)
    plt.show() 
