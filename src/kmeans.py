import numpy as np
import random

MAX_ITER = 30
EPS = 0.01

def update_colors(data, centroids, k):
    ## Iterate over all data points and find closest centroid. 
    ## Assigned color corresponds to closest centroid. 
    m,n = data.shape
    colors = np.zeros((m,1))
    for i in range(m):
        min_norm = float("inf")
        for j in range(k):
            norm = np.linalg.norm(data[i,:].toarray() - centroids[j,:], 2)
            if (norm < min_norm):
                min_norm = norm
                colors[i] = j
    return colors

def update_centroids(data, colors, k):
    ## Compute the centroid for each color.
    m,n = (data[1,:].toarray()).shape
    centroids = np.zeros((k,n))
    for i in range(k):
        indices = np.argwhere(colors == i)
        mu = np.sum(data[indices[:,0]], axis=0) / float(len(indices))
        centroids[i,:] = mu
    return centroids

def cluster(data, k):
    ## Randomly initialize centroids and alternate between
    ## assigning colors and updating centroids until convergence. 
    m,n = data.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        row_index = random.randint(0, (m-1))
        centroids[i,:] = data[row_index, :].toarray()
    
    iter = 0
    while (iter < MAX_ITER):
        print(iter)
        iter += 1
        colors = update_colors(data, centroids, k)
        centroids = update_centroids(data, colors, k)
    return centroids, colors
