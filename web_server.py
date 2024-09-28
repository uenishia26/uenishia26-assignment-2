from flask import Flask, render_template, request, jsonify
from app import generate_random_dataset, kmeans, kmeans_plus_plus, farthest_first, random, assign_clusters,update_centroids #Importing diffrent functions from app.py
import numpy as np
import json 
# Step 1: Create a Flask app
app = Flask(__name__)

#Defines the hompage 
@app.route('/')
def home(): #When user visits the homepage, it runs the homepage
    return render_template('index.html')

#Generating dataset 
@app.route('/generate', methods=['POST'])
def generate_data():
    X = generate_random_dataset()
    return jsonify(X.tolist())


@app.route('/initialize', methods=['POST'])
def initialize():
    # Get the input parameters from the front-end
    k = request.form.get('k')
    method = request.form.get('method')
    X_json = request.form.get('X')
    manual_centroids_json = request.form.get('manualCentroids')  # New

    if X_json is None:
        return jsonify({"error": "Dataset not provided!"}), 400

    X = np.array(json.loads(X_json))

    # Check if method is 'manual' and bypass the initialization
    if method == 'manual':
        # If manual centroids are provided, use them directly
        if manual_centroids_json is not None:
            centroids = np.array(json.loads(manual_centroids_json))
            return jsonify(centroids.tolist())
        else:
            return jsonify({"error": "Manual centroids not provided!"}), 400

    # Existing code for other methods
    if method == 'random':
        centroids = random(X, int(k))
    elif method == 'farthest_first':
        centroids = farthest_first(X, int(k))
    elif method == 'kmeans_plus_plus':
        centroids = kmeans_plus_plus(X, int(k))
    else:
        return jsonify({"error": "Invalid method!"}), 400

    # Return the initialized centroids to the front-end
    return jsonify(centroids.tolist())



@app.route('/step', methods=['POST'])
def step_through_kmeans():
    # Get the current centroids and dataset from the front-end
    centroids_json = request.form.get('centroids')
    X_json = request.form.get('X')

    if centroids_json is None or X_json is None:
        return jsonify({"error": "Centroids or dataset not provided!"}), 400

    # Convert JSON data into NumPy arrays
    centroids = np.array(json.loads(centroids_json))
    X = np.array(json.loads(X_json))

    # Step 1: Assign points to clusters
    clusters = assign_clusters(X, centroids)

    # Step 2: Update the centroids based on the new clusters
    new_centroids = update_centroids(X, clusters)

    # Step 3: Check for convergence
    diff = np.linalg.norm(new_centroids - centroids)
    has_converged = bool(diff < 1e-4)  # Set a small tolerance for convergence

    # Convert clusters and centroids to plain lists before returning
    clusters_as_lists = [[point.tolist() for point in cluster] for cluster in clusters]  # Convert each cluster
    centroids_as_lists = new_centroids.tolist()  # Convert NumPy array to list

    return jsonify({
        "new_centroids": centroids_as_lists,  # Return updated centroids
        "clusters": clusters_as_lists,  # Return updated clusters
        "converged": has_converged  # Return convergence status
    })

@app.route('/runToConvergence', methods=['POST'])
def runToConvergence():
    # Get the input parameters from the front-end
    k = request.form.get('k')
    method = request.form.get('method')
    X_json = request.form.get('X')
    centroids_json = request.form.get('centroids')  # Add centroids for manual case

    if X_json is None:
        return jsonify({"error": "Dataset not provided!"}), 400

    # Convert the dataset from JSON
    X = np.array(json.loads(X_json))

    # If method is manual, use provided centroids
    if method == 'manual' and centroids_json is not None:
        centroids = np.array(json.loads(centroids_json))
    else:
        centroids = None  # Set to None to initialize within the kmeans function

    # Call the kmeans function with centroids (manual or initialized)
    centroids, clusters = kmeans(X, int(k), centroids=centroids, method=method)

    # Convert clusters and centroids to plain lists before returning
    clusters_as_lists = [[point.tolist() for point in cluster] for cluster in clusters]
    centroids_as_lists = centroids.tolist()

    # Return the final centroids and clusters to the frontend
    return jsonify({
        "new_centroids": centroids_as_lists,
        "clusters": clusters_as_lists
    })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
