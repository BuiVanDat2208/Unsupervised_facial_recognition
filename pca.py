# pca.py File này sẽ thực hiện thuật toán PCA để giảm chiều dữ liệu và trích xuất đặc trưng.
from sklearn.decomposition import PCA

def apply_pca(X_scaled, n_components=150):
    # Áp dụng PCA để giảm chiều dữ liệu xuống còn n_components
    pca = PCA(n_components=n_components, whiten=True).fit(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")
    
    # Chuyển đổi dữ liệu vào không gian PCA mới
    X_pca = pca.transform(X_scaled)  # Dữ liệu giảm chiều
    
    return pca, X_pca
