import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping

from model import build_autoencoder
from utils import plot_reconstructions, visualize_compressed

# Load and preprocess
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full.astype("float32") / 255
X_train = X_train_full[:-5000].reshape(-1, 28 * 28)
X_valid = X_train_full[-5000:].reshape(-1, 28 * 28)
y_valid = y_train_full[-5000:]

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cluster_labels = [f"Cluster {i}" for i in range(10)]

# Build and train
input_dim = X_train.shape[1]
autoencoder, encoder = build_autoencoder(input_dim, latent_dim=100)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(X_train, X_train, epochs=50,
                          validation_data=(X_valid, X_valid),
                          callbacks=[early_stop], verbose=0)

# PCA
pca = PCA(n_components=100).fit(X_train)
X_valid_pca = pca.transform(X_valid)
X_valid_aed = encoder.predict(X_valid)

# t-SNE
X_valid_pca_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_valid_pca)
X_valid_aed_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_valid_aed)

# Clustering
kmeans_ae = KMeans(n_clusters=10, random_state=42, n_init='auto')
labels_ae = kmeans_ae.fit_predict(X_valid_aed)
kmeans_pca = KMeans(n_clusters=10, random_state=42, n_init='auto')
labels_pca = kmeans_pca.fit_predict(X_valid_pca)

# Scores
print(f"ARI (Autoencoder): {adjusted_rand_score(y_valid, labels_ae):.3f}")
print(f"ARI (PCA): {adjusted_rand_score(y_valid, labels_pca):.3f}")
print(f"Homogeneity (AE): {homogeneity_score(y_valid, labels_ae):.3f}")
print(f"Homogeneity (PCA): {homogeneity_score(y_valid, labels_pca):.3f}")
print(f"Silhouette (AE): {silhouette_score(X_valid_aed, labels_ae):.3f}")
print(f"Silhouette (PCA): {silhouette_score(X_valid_pca, labels_pca):.3f}")

# Visuals
plot_reconstructions(autoencoder, X_valid)
visualize_compressed(X_valid_pca_tsne, y_valid, X_valid, result="PCA Results", class_names=class_names)
visualize_compressed(X_valid_aed_tsne, y_valid, X_valid, result="Autoencoder Results", class_names=class_names)
visualize_compressed(X_valid_aed_tsne, labels_ae, X_valid, result="Autoencoder KMeans", class_names=cluster_labels)
visualize_compressed(X_valid_pca_tsne, labels_pca, X_valid, result="PCA KMeans", class_names=cluster_labels)
