from sklearn.datasets import fetch_openml
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Problem 02
mnist = fetch_openml('mnist_784', version = 1)
mnist.keys()
random.seed(0)

X = np.array(mnist['data'])
y = np.array(mnist['target'])
sample_X = []
sample_y = []

indices = {str(i): [] for i in range(10)}

for num, label in enumerate(y):
    indices[label].append(num)

choices = []
sample_size = 100

for label in indices:
    choices.extend(random.sample(indices[label], sample_size))

for index in choices:
    sample_X.append(X[index])
    sample_y.append(y[index])

sample_X = np.array(sample_X)
sample_y = np.array(sample_y)

print(sample_X.shape)
print(sample_y.shape)

# Problem 03

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(sample_X, n_neighbors=params["n_neighbors"], include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

#Agglomerative clustering
Agg_Clur = AgglomerativeClustering(n_clusters= 10, linkage="ward", connectivity=connectivity).fit(sample_X)

#K-means clustering
kmeans = KMeans(n_clusters=10).fit(sample_X)

#Gaussian Mixture
gaussian = GaussianMixture(n_components=10, covariance_type="full").fit(sample_X)

#Spectral clustering
spectral = SpectralClustering(n_clusters=10, eigen_solver = 'arpack', affinity='nearest_neighbors').fit(sample_X)

# Problem 04

rand_score_list = []

Agg_clur_rand_score = adjusted_rand_score(sample_y, Agg_Clur.labels_)
kmeans_rand_score = adjusted_rand_score(sample_y, kmeans.labels_)
guassian_rand_score = adjusted_rand_score(sample_y, gaussian.predict(sample_X))
spectral_rand_score = adjusted_rand_score(sample_y, spectral.labels_)

rand_score_list.extend([Agg_clur_rand_score, kmeans_rand_score, guassian_rand_score, spectral_rand_score])

plt.title("Rand Index")
plt.bar(['Agglomerative Clustering', 'K-means Clustering', 'Gaussian Mixture', 'Spectral Clustering'], rand_score_list, width=0.4, color="green")
plt.xticks(fontsize=8)
plt.xlabel('Models')
plt.ylim([0, 0.5])
plt.show()

print(rand_score_list)

mutual_score_list = []

Agg_clur_mutual_score = adjusted_mutual_info_score(sample_y, Agg_Clur.labels_)
kmeans_mutual_score = adjusted_mutual_info_score(sample_y, kmeans.labels_)
guassian_mutual_score = adjusted_mutual_info_score(sample_y, gaussian.predict(sample_X))
spectral_mutual_score = adjusted_mutual_info_score(sample_y, spectral.labels_)

mutual_score_list.extend([Agg_clur_mutual_score, kmeans_mutual_score, guassian_mutual_score, spectral_mutual_score])

plt.title("Mutual Information Based Score")
plt.bar(['Agglomerative Clustering', 'K-means Clustering', 'Gaussian Mixture', 'Spectral Clustering'], mutual_score_list, width=0.4, color="blue")
plt.xticks(fontsize=8)
plt.xlabel('Models')
plt.ylim([0, 0.65])
plt.show()

print(mutual_score_list)

# Problem 05
mutual_score_list = []
rand_index_list = []

# Calculate mutual information and Rand index for Agglomerative Clustering
clf = NearestCentroid()
clf.fit(sample_X, Agg_Clur.labels_)
neigh = KNeighborsClassifier(n_neighbors=1)
X = clf.centroids_
Y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
neigh.fit(X, Y)

mutual_score = adjusted_mutual_info_score(sample_y, neigh.predict(sample_X))
rand_index = adjusted_rand_score(sample_y, neigh.predict(sample_X))
rand_index_list.append(rand_index)
mutual_score_list.append(mutual_score)

# Calculate mutual information and Rand index for K-means Clustering
neigh = KNeighborsClassifier(n_neighbors=1)
X = kmeans.cluster_centers_
Y = kmeans.predict(X)
neigh.fit(X, Y)

mutual_score = adjusted_mutual_info_score(sample_y, neigh.predict(sample_X))
rand_index = adjusted_rand_score(sample_y, neigh.predict(sample_X))
rand_index_list.append(rand_index)
mutual_score_list.append(mutual_score)

# Calculate mutual information and Rand index for Gaussian Mixture
neigh = KNeighborsClassifier(n_neighbors=1)
X = gaussian.means_
Y = gaussian.predict(X)
neigh.fit(X, Y)

mutual_score = adjusted_mutual_info_score(sample_y, neigh.predict(sample_X))
rand_index = adjusted_rand_score(sample_y, neigh.predict(sample_X))
rand_index_list.append(rand_index)
mutual_score_list.append(mutual_score)

# Calculate mutual information and Rand index for Spectral Clustering
clf = NearestCentroid()
clf.fit(sample_X, spectral.labels_)
neigh = KNeighborsClassifier(n_neighbors=1)
X = clf.centroids_
Y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
neigh.fit(X, Y)

mutual_score = adjusted_mutual_info_score(sample_y, neigh.predict(sample_X))
rand_index = adjusted_rand_score(sample_y, neigh.predict(sample_X))
rand_index_list.append(rand_index)
mutual_score_list.append(mutual_score)

plt.title("Rand Index")
plt.bar(['Agglomerative 1NN ', 'K-means 1NN', 'Guassian Mixture 1NN', 'Spectral 1-NN'], rand_index_list, width=0.4, color="green")
plt.xticks(fontsize=8)
plt.xlabel('models')
plt.ylim([0, 0.5])
plt.show()
print(rand_index_list)

plt.title("Mutual Information Based Score")
plt.bar(['Agglomerative 1NN ', 'K-means 1NN', 'Guassian Mixture 1NN', 'Spectral 1-NN'], mutual_score_list, width=0.4, color="blue")
plt.xticks(fontsize=8)
plt.xlabel('Models')
plt.ylim([0, 0.65])
plt.show()
print(mutual_score_list)