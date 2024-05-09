import numpy as np


def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroid_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[centroid_indices]
    return centroids



def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = []
    centroids.append(data[np.random.randint(data.shape[0])])
    for _ in range(1, k):
        distances = []
        distancesIdx = []
        idx = 0
        for j,x in enumerate(data):
            min_distance = float('inf')
            for c in centroids:
                
                distance = np.linalg.norm(x - c)
                if distance < min_distance:
                    min_distance = distance
                    idx = j
            distances.append(min_distance)
            distancesIdx.append(idx)
        for i,distance in enumerate(distances):
            if(distance == max(distances)):
                idx = i
            np.array(distancesIdx)
        centroids.append(data[distancesIdx[idx]])
    return np.array(centroids)


def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroid, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    centroids = np.array([data[assignments == i].mean(axis=0) for i in range(len(np.unique(assignments)))])
    return centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  
        # print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): 
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
