import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import random
import kharmonic_means

MAX_ITER = 30

def load_data(datafile):
    data = load_svmlight_file(datafile)
    return data[0], data[1]

def update_colors(data, centroids, k):
    ## Iterate over all data points and find closest centroid. 
    ## Assigned color corresponds to closest centroid. 
    m,n = data.shape
    colors = np.zeros((m,1))
    for i in range(m):
        min_norm = float("inf")
        for j in range(k):
            norm = np.linalg.norm(data[i,:].toarray() - centroids[:,j], 2)
            if (norm < min_norm):
                min_norm = norm
                colors[i] = j
    return colors

def update_centroids(data, colors, k):
    ## Compute the centroid for each color.
    m,n = (data[1,:].toarray()).shape
    centroids = np.zeros((n,k))
    for i in range(k):
        indices = np.argwhere(colors == i)
        mu = np.sum(data[indices[:,0]], axis=0) / float(len(indices))
        centroids[:,i] = mu
    return centroids

def kmeans(data, k):
    ## Randomly initialize centroids and alternate between
    ## assigning colors and updating centroids until convergence. 
    m,n = data.shape
    centroids = np.zeros((n, k))
    for i in range(k):
        row_index = random.randint(0, (m-1))
        centroids[:,i] = data[row_index, :].toarray()
    
    iter = 0
    while (iter < MAX_ITER):
        print(iter)
        iter += 1
        colors = update_colors(data, centroids, k)
        centroids = update_centroids(data, colors, k)
    return centroids, colors

if __name__ == '__main__':
    filename = \
    '/Users/danielbyrnes/regent_install/' \
    'legion/language/project/Parallel_SVM/data/ijcnn1.tr'
    data, labels = load_data(filename)
    #print(data)
    #print(labels)
    
    ## Run k-means clustering.
    k = 8
    #kharmonic_means.cluster(data, k)
    #exit()
    centroids, colors = kmeans(data, k)
    colors = np.ravel(colors)
    plt.hist(colors)
    
    plt.figure()
    lda = LDA(n_components = 2)
    lda_trans = pd.DataFrame(lda.fit_transform(data.toarray(), colors))
    plt.scatter(lda_trans[colors==0][0], lda_trans[colors==0][1], label='Class 1', c='pink') 
    plt.scatter(lda_trans[colors==1][0], lda_trans[colors==1][1], label='Class 2', c='red') 
    plt.scatter(lda_trans[colors==2][0], lda_trans[colors==2][1], label='Class 3', c='blue') 
    plt.scatter(lda_trans[colors==3][0], lda_trans[colors==3][1], label='Class 4', c='green') 
    plt.scatter(lda_trans[colors==4][0], lda_trans[colors==4][1], label='Class 5', c='yellow') 
    plt.scatter(lda_trans[colors==5][0], lda_trans[colors==5][1], label='Class 6', c='orange') 
    plt.scatter(lda_trans[colors==6][0], lda_trans[colors==6][1], label='Class 7', c='black') 
    plt.scatter(lda_trans[colors==7][0], lda_trans[colors==7][1], label='Class 8', c='purple') 
    plt.legend(loc=2)

    plt.show()


