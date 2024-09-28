import numpy as np

#Generating random Dataset
def generate_random_dataset(n_samples=100, n_features=2, random_state=None):
    np.random.seed(random_state) 
    X = 10 * np.random.randn(n_samples, n_features)
    return X


#Diffrent intialization methods 
#1: K means ++
def kmeans_plus_plus(X, k): 

    centroids = []

    #Randomly choose the first centroid 
    centroids.append(X[np.random.choice(X.shape[0])])

    for _ in range(1,k):
        #Finding distance to closest centroid for each pair of data points x,y
        distances = np.min([np.linalg.norm(X - centroid, axis=1)for centroid in centroids],0) 

        #Equation to assign probabilities 
        #Squared distance / sum of squared distances 
        probabilities = distances**2 / np.sum(distances**2) 

        new_idx_centroids = np.random.choice(X.shape[0], p=probabilities)
        centroids.append(X[new_idx_centroids])

    return np.array(centroids)

#2 Farthest First 
def farthest_first(X,k): 
    centroids = []

    #Randomly selecting first centroid 
    centroids.append(X[np.random.choice(X.shape[0])])

    for _ in range(1,k): 
        #How far a datapoint is from its nearest centroid 
        distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids],axis=0)
        next_centroid_idx = np.argmax(distances)
        centroids.append(X[next_centroid_idx])


    return np.array(centroids)


#Step 3: Random intialization 
def random(X, k):
    centroids = []
    randRows = np.random.choice(X.shape[0],k,replace=False) #Selecting random rows 
    for i in range(len(randRows)):  #Iterating through rows 
        centroids.append(X[randRows[i]])  #Append x,y pairs into centorids 
    return np.array(centroids)  #Returning a numpy array for efficent calculations 


#The backbone of the Kmeans algorithm 
#Step 1: Assigning the data points to their respective clusters 
def assign_clusters(X, centroids): 
    #This array is parallel to the centroids array (Aka items in i clusters repersents a point closest to the i centroid)
    clusters = [[] for _ in range(len(centroids))] 

    for idx, point in enumerate(X): 
        distances = [np.linalg.norm(point-centroid) for centroid in centroids]
        closestCentroid = np.argmin(distances)
        clusters[closestCentroid].append(point) #Convert numpy array to a normal list 

    return clusters


#Step 2: Updating the centroids 
def update_centroids(X, clusters):
    new_centroids = []
    #For each cluster, this basically adds up all the rows, columns and finds the mean of both row and colum 
    for cluster in clusters:
        updated_centroid = np.mean(cluster, axis=0) #axis = 0 means look at columns 
        new_centroids.append(updated_centroid)
    
    return np.array(new_centroids)


#Step 3: KMeans Algorithm 
def kmeans(X, k, centroids=None, method='random', max_iter=100, tolerance=1e-4):
    # Step 1: Initialize the centroids if not provided
    if centroids is None:
        if method == 'random':
            centroids = random(X, k)
        elif method == 'kmeans_plus_plus':
            centroids = kmeans_plus_plus(X, k)
        elif method == 'farthest_first':
            centroids = farthest_first(X, k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Step 2: Iterate the KMeans algorithm until convergence or max iterations
    for _ in range(max_iter):
        # Assign points to the nearest centroid
        clusters = assign_clusters(X, centroids)
        
        # Update centroids based on the mean of assigned clusters
        new_centroids = update_centroids(X, clusters)

        # Calculate the difference between new and old centroids
        diff = np.linalg.norm(new_centroids - centroids)

        # If the difference is smaller than the tolerance, we have convergence
        if diff < tolerance:
            break
        
        # Otherwise, update the centroids for the next iteration
        centroids = new_centroids
    
    return centroids, clusters  # Return final centroids and their clusters
