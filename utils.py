# utils.py File này chứa các hàm hỗ trợ để hiển thị kết quả.
import matplotlib.pyplot as plt

def plot_pca_variance(pca):
    """
    Vẽ biểu đồ thể hiện tỉ lệ biến thiên được giải thích bởi các thành phần chính trong PCA.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.xlabel('Component Number')
    plt.ylabel('Variance Explained')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.show()

def plot_kmeans_clusters(X_pca, y_kmeans, n_clusters=10):
    """
    Vẽ biểu đồ phân nhóm các khuôn mặt sau khi áp dụng K-means clustering.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o')
    plt.title(f"K-means Clustering with {n_clusters} Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster Label')
    plt.show()

def plot_gallery(images, titles, h, w, n_row=5, n_col=5):
    """
    Hàm này hiển thị các hình ảnh theo các nhóm phân loại.
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90)
    
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def show_results(images, y_kmeans, n_clusters, target_names):
    """
    Hiển thị các kết quả phân nhóm khuôn mặt theo từng nhóm của K-means.
    """
    titles = [f"Cluster {i + 1}" for i in range(len(y_kmeans))]
    plot_gallery(images, titles, images.shape[1], images.shape[2])
    plt.show()
