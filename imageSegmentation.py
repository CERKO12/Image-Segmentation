#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
- Author: <Congkai Sun>

'''


# In[ ]:


import numpy as np
from PIL import Image
import sys
from sklearn.cluster import KMeans
import math


# In[2]:


def load_image(filename):
    """
    Load the image file into a numpy array
    """
    image = Image.open(filename)
    return np.array(image)


# In[3]:


def standardize_data(data):
    """
    Standardize the data
    """
    data = data.astype(np.float64)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


# In[4]:


def unstandardize_data(data, original_data):
    """
    Unstandardize the data using the original data to get the original values
    """
    mean = np.mean(original_data, axis=0)
    std = np.std(original_data, axis=0)
    return (data * std) + mean


# In[5]:


def coordinate_descent_kmeans(X, k, max_iter=100, tol=1e-4):
    """
    Coordinate descent algorithm for k-means clustering.

    Parameters:
        X (array-like): data matrix with shape (n_samples, n_features)
        k (int): number of clusters
        max_iter (int): maximum number of iterations (default: 100)
        tol (float): convergence tolerance (default: 1e-4)

    Returns:
        centroids (array-like): cluster centroids with shape (k, n_features)
        clusters (array-like): cluster assignments for each data point with shape (n_samples,)
    """
    
    # Initialize cluster centroids randomly from the data points
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False), :]
    
    for i in range(max_iter):
        clusters = np.zeros(n_samples)
        for j, x in enumerate(X):
            # Assign each point to the closest centroid by Euclidean distance
            distances = np.linalg.norm(x - centroids, axis = 1)
            clusters[j] = np.argmin(distances)
        prev_centroids = np.copy(centroids)
        for c in range(k):
            # The mean of the points assigned to it
            centroids[c, :] = np.mean(X[clusters == c, :], axis=0)
        # Check the sum of absolute differences between the current and previous centroids.
        if np.sum(np.abs(centroids - prev_centroids)) < tol:
            break
    return centroids, clusters


# In[6]:


def segment_image(image, labels, centers):
    """
    Segment the image
    """
    new_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = int(labels[i * image.shape[1] + j])
            new_image[i, j, :] = centers[label, :3]
    return new_image


# In[7]:


def save_image(filename, image):
    """
    Save the image to the specified filename
    """
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(filename)


# In[8]:


def main():
    # Command line arguments
    k = int(sys.argv[1])
    inputImageFilename = sys.argv[2]
    outputImageFilename = sys.argv[3]
    
    # Load image and convert to data
    image = load_image(inputImageFilename)
    data = np.zeros((image.shape[0] * image.shape[1], 5))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            data[i * image.shape[1] + j, 0] = image[i, j, 0]
            data[i * image.shape[1] + j, 1] = image[i, j, 1]
            data[i * image.shape[1] + j, 2] = image[i, j, 2]
            data[i * image.shape[1] + j, 3] = i
            data[i * image.shape[1] + j, 4] = j

    # Standardize data
    standardized_data = standardize_data(data)

    # Apply k-means clustering
    centers, labels = coordinate_descent_kmeans(standardized_data, k)

    # Segment the image
    new_image = segment_image(image, labels, unstandardize_data(centers, data))

    # Save the image
    save_image(outputImageFilename, new_image)

if __name__ == '__main__':
    main()


# In[ ]:




