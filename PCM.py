# -*- coding: utf-8 -*-
"""
File:  PCM.py
Atthor:  Tashfique Hasnine Choudhury
Date:  11/08/2021 
Desc:  Possibilistic C Means  
    
"""


""" =======================  import dependencies ========================== """
"""
Code adapted from: https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/cluster/_cmeans.py
"""
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import datasets 

plt.close('all') #close any open plots


""" ======================  Function definitions ========================== """
#Add function definitions here
def Pcm0(data, t_old, c, q):

    # Normalizing, then eliminating any potential zero values.
    t_old /= np.ones((c, 1)).dot(np.atleast_2d(t_old.sum(axis=0)))
    t_old = np.fmax(t_old, np.finfo(np.float64).eps)

    tq = t_old ** q
    
    # Calctlate cluster centers
    data = data.T
    cntr = tq.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(tq.sum(axis=1))).T)

    d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)
    
    nta = np.sum(tq * d ** 2, axis=1) / np.sum(tq, axis=1) #Eta
    
    jm = (tq * d ** 2).sum() + nta.sum() * ((1 - t_old) ** q).sum() #Objective func.
    #PCM equation
    t = (1 + (((d **2).T / nta).T **(1 / (q - 1)))) **-1

    t /= np.ones((c, 1)).dot(np.atleast_2d(t.sum(axis=0)))

    return cntr, t, jm, d

def _distance(data, centers):
    return cdist(data, centers).T

def cmeans(data, c, q, error=1e-3, maxiter=300, init=None, seed=None):
    # Setup t0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        t0 = np.random.rand(c, n)
        t0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(t0.sum(axis=0))).astype(np.float64)
        init = t0.copy()
    t0 = init
    t = np.fmax(t0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        t2 = t.copy()
        [cntr, t, Jjm, d] = Pcm0(data, t2, c, q)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(t - t2) < error:
            break

    # Final calculations
    error = np.linalg.norm(t - t2)

    return cntr, t
        
""" ======================  Variable Declaration ========================== """
#Set or load parameters here
n_samples = 1500
n_clusters = 3
q = 2

""" =======================  Generate Data ======================= """

#Blobs
blobs, y_blobs = datasets.make_blobs(n_samples=n_samples)
X = blobs
y = y_blobs

# Anisotropicly distributed blobs
transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(blobs, transformation)
y_aniso = y_blobs

# Different variance blobs 
X_varied, y_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    
""" ========================  Cluster the Data ============================= """

centers, L = cmeans(X.T, n_clusters, q)
centers, L_aniso = cmeans(X_aniso.T, n_clusters,q)
centers, L_varied = cmeans(X_varied.T, n_clusters,q)
centers, L_filtered = cmeans(X_filtered.T, n_clusters,q)

""" ========================  Plot Results ============================== """


plt.figure(figsize=(12, 12))
plt.subplot(431)
plt.scatter(X[:, 0], X[:, 1], c=L[0,:])
plt.subplot(432)
plt.scatter(X[:, 0], X[:, 1], c=L[1,:])
plt.title("Blobs")
plt.subplot(433)
plt.scatter(X[:, 0], X[:, 1], c=L[2,:])

plt.subplot(434)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=L_aniso[0,:])
plt.subplot(435)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=L_aniso[1,:])
plt.title("Anisotropicly Distributed Blobs")
plt.subplot(436)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=L_aniso[2,:])

plt.subplot(437)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=L_varied[0,:])
plt.subplot(438)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=L_varied[1,:])
plt.title("Unequal Variance - Only Blobs")
plt.subplot(439)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=L_varied[2,:])

plt.subplot(4,3,10)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=L_filtered[0,:])
plt.subplot(4,3,11)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=L_filtered[1,:])
plt.title("Unevenly Sized Blobs")
plt.subplot(4,3,12)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=L_filtered[2,:])

