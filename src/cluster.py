import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import random
import kmeans
import kharmonic_means

def load_data(datafile):
    data = load_svmlight_file(datafile)
    return data[0], data[1]

if __name__ == '__main__':
    clustered = True
    k = 8

    filename = \
    '/Users/danielbyrnes/regent_install/' \
    'legion/language/project/Parallel_SVM/data/ijcnn1.tr'
    data, labels = load_data(filename)

    ''' Random partition code. 
    m, n = data.shape 
    random_partition = np.floor(np.random.uniform(0, k, m))
    print(random_partition)
    print(random_partition.shape)
    plt.hist(random_partition)
    plt.show()
    np.savetxt('partitions/random_partition.tr', random_partition, '%1.0f')

    test_filename = \
    '/Users/danielbyrnes/regent_install/' \
    'legion/language/project/Parallel_SVM/data/ijcnn1.t'
    test_data, test_labels = load_data(test_filename)
    m, n = test_data.shape 
    random_partition = np.floor(np.random.uniform(0, k, m))
    np.savetxt('partitions/random_partition.t', random_partition, '%1.0f')
    '''

    if not clustered:
        ## Run k-means clustering.
        #kharmonic_means.cluster(data, k)
        centroids, colors = kmeans.cluster(data, k)
        colors = np.ravel(colors)
        np.savetxt('partitions/kmeans_centroids_30.tr', centroids, '%1.4e')
        np.savetxt('partitions/kmeans_clustering_30.tr', colors, '%1.0f')
    else:
        colors = np.loadtxt('partitions/kmeans_clustering_30.tr')
        centroids = np.loadtxt('partitions/kmeans_centroids_30.tr')
        #colors = np.loadtxt('partitions/kernel_kmeans_partition.tr')

    plt.hist(colors)
    plt.xlabel('Cluster #')
    plt.ylabel('# Instances')
    plt.title('Histogram for Kernel Kmeans Clustering')
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
    plt.legend(loc=4)

    '''
    test_filename = \
    '/Users/danielbyrnes/regent_install/' \
    'legion/language/project/Parallel_SVM/data/ijcnn1.t'
    test_data, test_labels = load_data(test_filename)
    # Centroids are ordered by 'color' assignment.
    # Assign test data instances to the nearest centroid.
    colors = kmeans.update_colors(test_data, centroids, k)
    colors = np.ravel(colors)
    np.savetxt('partitions/kmeans_clustering_30.t', colors, '%1.0f')
    '''  

    plt.figure()
    pca = PCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(data.toarray()))
    plt.scatter(transformed[colors==0][0], transformed[colors==0][1], label='Class 1', c='pink') 
    plt.scatter(transformed[colors==1][0], transformed[colors==1][1], label='Class 2', c='red') 
    plt.scatter(transformed[colors==2][0], transformed[colors==2][1], label='Class 3', c='blue') 
    plt.scatter(transformed[colors==3][0], transformed[colors==3][1], label='Class 4', c='green') 
    plt.scatter(transformed[colors==4][0], transformed[colors==4][1], label='Class 5', c='yellow') 
    plt.scatter(transformed[colors==5][0], transformed[colors==5][1], label='Class 6', c='orange') 
    plt.scatter(transformed[colors==6][0], transformed[colors==6][1], label='Class 7', c='black') 
    plt.scatter(transformed[colors==7][0], transformed[colors==7][1], label='Class 8', c='purple') 
    plt.show()







   
