# kmeans.py File này thực hiện thuật toán K-means và  Spectral Clustering  để phân nhóm các khuôn mặt.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

def apply_kmeans(X_pca, n_clusters=10):
    """
    Áp dụng K-means để phân nhóm khuôn mặt vào n_clusters nhóm.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_pca)
    print(f"K-means Cluster centers: {kmeans.cluster_centers_}")
    
    y_kmeans = kmeans.predict(X_pca)
    
    return kmeans, y_kmeans


def apply_spectral_clustering(X_pca, n_clusters=10):
    """
    Áp dụng Spectral Clustering để phân nhóm khuôn mặt vào n_clusters nhóm.
    """
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    y_spectral = spectral.fit_predict(X_pca)
    
    print(f"Spectral Clustering labels: {np.unique(y_spectral)}")
    
    return spectral, y_spectral
