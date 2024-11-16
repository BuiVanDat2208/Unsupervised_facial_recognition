# data_loader.py File này sẽ xử lý việc tải và tiền xử lý dữ liệu.
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler

def load_data():
    # Tải dữ liệu LFW (Labeled Faces in the Wild)
    lfw_data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    print(f"Dataset loaded with {lfw_data.images.shape[0]} images and {lfw_data.images.shape[1]} x {lfw_data.images.shape[2]} resolution.")
    
    # Lấy ảnh và nhãn (target) từ dataset
    X = lfw_data.data  # Dữ liệu ảnh (dạng vector)
    y = lfw_data.target  # Nhãn của khuôn mặt
    images = lfw_data.images  # Dữ liệu hình ảnh gốc
    
    # Chuẩn hóa dữ liệu (standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Chuẩn hóa các giá trị ảnh để chúng có cùng phạm vi
    
    return X_scaled, y, images, lfw_data.target_names
