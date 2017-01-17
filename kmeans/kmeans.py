import kmeanslib as lib
import tensorflow as tf

n_features = 2
n_clusters = 3
n_samples_per_cluster = 50
seed = 700
embiggen_factor = 70
centroids, samples = lib.create_samples(
    n_clusters,
    n_samples_per_cluster,
    n_features,
    embiggen_factor,
    seed)
initial_centroids = lib.choose_random_centroids(samples, n_clusters)

model = tf.initialize_all_variables()
with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(initial_centroids)
    # centroid_values = session.run(centroids)

# plot_clusters(sample_values, centroid_values, n_samples_per_cluster)



data_centroids, samples = lib.create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = lib.choose_random_centroids(samples, n_clusters)
nearest_indices = lib.assign_to_nearest(samples, initial_centroids)
updated_centroids = lib.update_centroids(samples, nearest_indices, n_clusters)

model = tf.initialize_all_variables()
with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)

lib.plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)