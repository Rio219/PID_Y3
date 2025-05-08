import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
# Hàm padding ảnh để kích thước chia hết cho block_size
def pad_image(image, block_size):
    H, W = image.shape[:2]
    pad_H = (block_size - H % block_size) % block_size
    pad_W = (block_size - W % block_size) % block_size
    padded = np.pad(image, ((0, pad_H), (0, pad_W), (0, 0)), mode='edge')
    return padded, H, W

# Hàm chọn 1 ảnh từ máy
def select_image():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ gốc Tkinter

    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để nén PCA",
        filetypes=[
            ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Tất cả file", "*.*")
        ]
    )
    return file_path if file_path else None
    
    return list(file_paths)
def save_output_image(image_bgr, original_path, k):
    """
    Lưu ảnh đã nén theo định dạng [ten_anh_goc]_pca_k[so_components].jpg vào thư mục outputs/
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Thư mục chứa file .py
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(original_path)
    name, _ = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_pca_k{k}.jpg")
    cv2.imwrite(output_path, image_bgr)
    print(f"Ảnh đã được lưu tại: {output_path}")

# Hàm nén và tái tạo một kênh màu
def compress_channel(channel, block_size, k):
    H_padded, W_padded = channel.shape
    blocks = []
    # Lấy từng khối và làm phẳng
    for i in range(0, H_padded, block_size):
        for j in range(0, W_padded, block_size):
            block = channel[i:i+block_size, j:j+block_size].flatten()
            blocks.append(block)
    blocks = np.array(blocks)
    
    # Áp dụng PCA
    pca = PCA(n_components=k)
    blocks_pca = pca.fit_transform(blocks)
    blocks_reconstructed = pca.inverse_transform(blocks_pca)
    
    # Tái tạo kênh từ các khối
    channel_reconstructed = np.zeros((H_padded, W_padded))
    idx = 0
    for i in range(0, H_padded, block_size):
        for j in range(0, W_padded, block_size):
            block = blocks_reconstructed[idx].reshape(block_size, block_size)
            channel_reconstructed[i:i+block_size, j:j+block_size] = block
            idx += 1
    return channel_reconstructed



# Hàm chính
def main(image_path, block_size, k):
    # Đọc và chuyển ảnh sang RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Padding ảnh
    padded_image, original_H, original_W = pad_image(image_rgb, block_size)
    
    # Tách thành 3 kênh
    R, G, B = padded_image[:,:,0], padded_image[:,:,1], padded_image[:,:,2]
    
    # Nén và tái tạo từng kênh
    R_reconstructed = compress_channel(R, block_size, k)
    G_reconstructed = compress_channel(G, block_size, k)
    B_reconstructed = compress_channel(B, block_size, k)
    
    # Ghép các kênh thành ảnh RGB
    reconstructed_padded = np.stack([R_reconstructed, G_reconstructed, B_reconstructed], axis=2)
    
    # Cắt bỏ padding để trở về kích thước gốc
    reconstructed = reconstructed_padded[:original_H, :original_W, :]
    
    # Chuẩn hóa giá trị pixel về [0, 255] và chuyển sang uint8
    reconstructed_clipped = np.clip(reconstructed, 0, 255)
    reconstructed_uint8 = np.round(reconstructed_clipped).astype(np.uint8)
    
    # Tính MSE và PSNR
    original = image_rgb
    mse = np.mean((original.astype(float) - reconstructed_uint8.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    
    # Lưu ảnh tái tạo (chuyển sang BGR vì OpenCV yêu cầu)
    reconstructed_bgr = cv2.cvtColor(reconstructed_uint8, cv2.COLOR_RGB2BGR)
    save_output_image(reconstructed_bgr, image_path, k)    
    # Hiển thị ảnh
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Ảnh gốc')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_uint8)
    plt.title('Ảnh nén')
    plt.axis('off')
    plt.show()
    
    # In kết quả
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")

# Sử dụng ví dụ
image_path = select_image()
block_size = 8  # Kích thước khối, ví dụ 8x8
k = 20         # Số thành phần chính giữ lại
main(image_path, block_size, k)