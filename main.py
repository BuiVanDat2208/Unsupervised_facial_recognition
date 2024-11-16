# main.py File này là điểm bắt đầu của chương trình, nơi ta kết hợp các thuật toán PCA, K-means và  Spectral Clustering.
from data_loader import load_data
from pca import apply_pca
from kmeans import apply_kmeans, apply_spectral_clustering

from utils import show_results, plot_pca_variance, plot_kmeans_clusters

def main():
    # Tải dữ liệu
    X_scaled, y, images, target_names = load_data()

    # Áp dụng PCA để giảm chiều dữ liệu
    pca, X_pca = apply_pca(X_scaled)

    # Vẽ biểu đồ độ phân tán của các thành phần chính (PCA)
    plot_pca_variance(pca)

    # Áp dụng K-means để phân nhóm khuôn mặt
    kmeans, y_kmeans = apply_kmeans(X_pca, n_clusters=10)
    
    spectral_clustering, y_kmeans = apply_spectral_clustering(X_pca, n_clusters=10)

    # Vẽ biểu đồ phân nhóm K-means
    plot_kmeans_clusters(X_pca, y_kmeans, n_clusters=10)

    # Hiển thị kết quả phân nhóm (các khuôn mặt được phân vào các nhóm)
    show_results(images, y_kmeans, n_clusters=10, target_names=target_names)

if __name__ == '__main__':
    main()
