#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
# Lấy thư mục chứa file script Python hiện tại
script_dir = os.path.dirname(os.path.abspath(__file__))

# Tạo thư mục outputs nếu chưa tồn tại
output_dir = os.path.join(script_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

class PanoramaStitcher:
    """
    Lớp chính xử lý việc ghép ảnh panorama sử dụng SIFT, Homography và Blending
    """
    def __init__(self, max_features=10000, ratio_thresh=0.7, reproj_thresh=4.0):
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        self.reproj_thresh = reproj_thresh
        
    def resize_image_if_large(self, img, max_dim=2000):
        """
        Thay đổi kích thước ảnh nếu quá lớn để tăng hiệu suất
        """
        h, w = img.shape[:2]
        
        # Nếu ảnh nhỏ hơn ngưỡng, giữ nguyên
        if max(h, w) <= max_dim:
            return img, 1.0
        
        # Tính tỉ lệ giảm kích thước
        scale = max_dim / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        
        # Thay đổi kích thước ảnh
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        print(f"Đã thay đổi kích thước ảnh từ {w}x{h} thành {new_size[0]}x{new_size[1]}")
        
        return resized, scale
    
    def enhance_image(self, img):
        """
        Áp dụng các bộ lọc nâng cao chất lượng ảnh
        """
        result = img.copy()
        
        # Tăng độ tương phản
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Làm sắc nét
        kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel)
        
        return result
    
    def apply_multiband_blending(self, img1, img2, mask, levels=6):
        """
        Áp dụng kỹ thuật multiband blending cho vùng chồng lấp
        """
        # Tạo pyramid cho mask
        mask_pyr = [mask]
        for i in range(levels):
            mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))
        
        # Tạo Laplacian pyramid cho ảnh
        lap_pyr1 = []
        lap_pyr2 = []
        
        # Tạo Gaussian pyramid cho ảnh gốc
        curr1 = img1.astype(np.float32)
        curr2 = img2.astype(np.float32)
        
        for i in range(levels):
            next1 = cv2.pyrDown(curr1)
            next2 = cv2.pyrDown(curr2)
            
            # Mở rộng ảnh lớp tiếp theo và tính Laplacian
            expanded = cv2.pyrUp(next1, dstsize=(curr1.shape[1], curr1.shape[0]))
            lap1 = curr1 - expanded
            
            expanded = cv2.pyrUp(next2, dstsize=(curr2.shape[1], curr2.shape[0]))
            lap2 = curr2 - expanded
            
            lap_pyr1.append(lap1)
            lap_pyr2.append(lap2)
            
            curr1 = next1
            curr2 = next2
        
        # Thêm mức cuối cùng
        lap_pyr1.append(curr1)
        lap_pyr2.append(curr2)
        
        # Blend từng mức của pyramid
        blended_pyr = []
        for l1, l2, m in zip(lap_pyr1, lap_pyr2, mask_pyr):
            # Mở rộng mask cho phù hợp với kích thước channels
            m_expanded = np.expand_dims(m, axis=2) if m.ndim == 2 else m
            m_expanded = m_expanded / 255.0 if m_expanded.dtype == np.uint8 else m_expanded
            
            # Blend
            blended = l1 * m_expanded + l2 * (1 - m_expanded)
            blended_pyr.append(blended)
        
        # Tái cấu trúc từ pyramid
        blended = blended_pyr[-1]
        for lap in reversed(blended_pyr[:-1]):
            blended = cv2.pyrUp(blended, dstsize=(lap.shape[1], lap.shape[0]))
            blended += lap
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def detect_and_compute_keypoints(self, img_gray):
        """
        Phát hiện và tính toán các keypoints và descriptors sử dụng SIFT
        
        Args:
            img_gray: Ảnh grayscale đầu vào
            
        Returns:
            keypoints, descriptors
        """
        sift = cv2.SIFT_create(self.max_features)
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)
        return keypoints, descriptors
    
    def match_keypoints(self, des1, des2):
        """
        Khớp các keypoints dựa trên descriptors và lọc sử dụng Lowe's ratio test
        
        Args:
            des1, des2: SIFT descriptors của hai ảnh
            
        Returns:
            good_matches: Danh sách các matches tốt sau khi lọc
        """
        # Sử dụng BFMatcher với norm L2 cho SIFT
        bf = cv2.BFMatcher(cv2.NORM_L2)
        
        # Tìm 2 matches gần nhất cho mỗi descriptor
        raw_matches = bf.knnMatch(des1, des2, k=2)
        
        # Áp dụng Lowe's ratio test
        good_matches = []
        for m, n in raw_matches:
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)
                
        return good_matches, raw_matches
    
    def compute_homography(self, kp1, kp2, good_matches):
        """
        Tính ma trận homography giữa hai ảnh sử dụng RANSAC
        
        Args:
            kp1, kp2: Keypoints của hai ảnh
            good_matches: Các matches đã được lọc
            
        Returns:
            H: Ma trận homography 3x3
            mask: Mặt nạ chỉ ra các inliers
        """
        if len(good_matches) < 4:
            raise RuntimeError("Không đủ good matches để tính homography (cần ít nhất 4)")
            
        # Tạo mảng các điểm tương ứng
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Tính homography sử dụng RANSAC
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, self.reproj_thresh)
        
        return H, mask
    
    def create_smoothing_mask(self, img1, img2, overlap_region, smoothing_window_percent=0.15):
        """
        Tạo mask làm mịn cho việc blend hai ảnh
        
        Args:
            img1, img2: Hai ảnh cần blend
            overlap_region: Vùng chồng lấp
            smoothing_window_percent: Phần trăm kích thước cửa sổ làm mịn
            
        Returns:
            mask: Mặt nạ làm mịn
        """
        width1 = img1.shape[1]
        width2 = img2.shape[1]
        lowest_width = min(width1, width2)
        smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000))
        
        # Xác định vị trí đường ghép
        seam_x = overlap_region[0] + overlap_region[2] // 2
        
        # Kích thước của cửa sổ làm mịn
        offset = int(smoothing_window_size / 2)
        
        # Chiều cao và rộng của mask
        h, w = img1.shape[:2]
        
        # Tạo mask cho blend mịn
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Tạo gradient từ 0 đến 1 trong vùng chuyển tiếp
        # Bên trái sẽ có giá trị 1, bên phải sẽ có giá trị 0
        left_edge = max(0, seam_x - offset)
        right_edge = min(w, seam_x + offset)
        
        # Đảm bảo rằng chúng ta có một gradient mịn trong vùng chuyển tiếp
        if right_edge > left_edge:
            x = np.arange(left_edge, right_edge)
            mask_col = np.zeros(h)
            
            for i in range(left_edge, right_edge):
                # Tính giá trị gradient từ 1->0
                alpha = 1.0 - ((i - left_edge) / (right_edge - left_edge))
                mask[:, i] = alpha
                
            # Đặt giá trị 1 cho phần bên trái
            mask[:, :left_edge] = 1.0
            
        return cv2.merge([mask, mask, mask])
    
    def stitch_images(self, img1, img2, show_matches=False):
        """
        Ghép hai ảnh thành ảnh panorama
        
        Args:
            img1: Ảnh đích (sẽ được giữ nguyên)
            img2: Ảnh sẽ được warp vào ảnh đích
            show_matches: Hiển thị các điểm trùng khớp
            
        Returns:
            result: Ảnh panorama đã ghép
            matches_image: Ảnh hiển thị các matches (hoặc None nếu không yêu cầu)
        """
        # Kiểm tra ảnh đầu vào
        if img1 is None or img2 is None:
            raise ValueError("Không thể đọc một trong các ảnh đầu vào")
        
        # Thay đổi kích thước ảnh nếu quá lớn để tăng hiệu suất
        img1, scale1 = self.resize_image_if_large(img1)
        img2, scale2 = self.resize_image_if_large(img2)
        
        # Chuyển ảnh sang grayscale cho việc tìm keypoints
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 1. Phát hiện và tính toán keypoints
        print("Đang phát hiện keypoints và tính descriptors bằng SIFT...")
        kp1, des1 = self.detect_and_compute_keypoints(img1_gray)
        kp2, des2 = self.detect_and_compute_keypoints(img2_gray)
        print(f"Tìm thấy {len(kp1)} keypoints trong ảnh 1 và {len(kp2)} keypoints trong ảnh 2")
        
        # 2. Khớp keypoints
        print("Đang khớp các điểm đặc trưng bằng Lowe's ratio test...")
        good_matches, raw_matches = self.match_keypoints(des1, des2)
        print(f"Số lượng good matches: {len(good_matches)}/{len(raw_matches)}")
        
        # 3. Tính homography
        H, mask = self.compute_homography(kp1, kp2, good_matches)
        inliers = mask.sum()
        print(f"Số lượng inliers: {inliers}/{len(good_matches)} ({inliers/len(good_matches)*100:.1f}%)")
        
        # 4. Hiển thị matches nếu được yêu cầu
        matches_image = None
        if show_matches:
            matches_mask = mask.ravel().tolist()
            matches_image = cv2.drawMatches(
                img1, kp1, img2, kp2, good_matches, None,
                matchColor=(0, 255, 0), singlePointColor=None,
                matchesMask=matches_mask,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        
        # 5. Tính kích thước canvas panorama
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Tính toán góc của ảnh thứ 2 sau khi transform
        corners2 = np.float32([[0, 0], [w2-1, 0], [w2-1, h2-1], [0, h2-1]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H)
        
        # Kết hợp với các góc của ảnh 1 để xác định kích thước canvas
        all_corners = np.vstack((
            warped_corners2, 
            np.float32([[0, 0], [w1-1, 0], [w1-1, h1-1], [0, h1-1]]).reshape(-1, 1, 2)
        ))
        
        # Tính toán giới hạn của canvas với biên an toàn
        x_min, y_min = np.floor(all_corners.min(axis=0).ravel() - 0.5).astype(int)
        # Sau khi ceil, +1 để bao phủ đủ pixel cuối
        x_max, y_max = (np.ceil(all_corners.max(axis=0).ravel() + 0.5) + 1).astype(int)
        
        # Vector dịch chuyển để đưa tất cả nội dung vào canvas dương
        trans = [-x_min, -y_min]
        H_trans = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]])
        
        # 6. Tính toán kích thước canvas và warp ảnh 2
        canvas_w, canvas_h = x_max - x_min, y_max - y_min
        print(f"Kích thước panorama: {canvas_w}x{canvas_h}")
        
        # Tạo ảnh warped của ảnh 2
        warped2 = cv2.warpPerspective(img2, H_trans.dot(H), (canvas_w, canvas_h))
        
        # 7. Ghép hai ảnh
        # Đặt ảnh 1 vào vị trí
        panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        panorama[trans[1]:trans[1]+h1, trans[0]:trans[0]+w1] = img1
        
        # 8. Tạo mask cho việc blending
        # Tìm vùng chồng lấp
        img1_region = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        img1_region[trans[1]:trans[1]+h1, trans[0]:trans[0]+w1] = 255
        
        img2_warped_gray = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)
        _, img2_region = cv2.threshold(img2_warped_gray, 1, 255, cv2.THRESH_BINARY)
        
        overlap = cv2.bitwise_and(img1_region, img2_region)
        
        # Tìm kích thước và vị trí của vùng chồng lấp
        overlap_contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Nếu có vùng chồng lấp
        if overlap_contours and np.sum(overlap) > 0:
            # Lấy contour lớn nhất của vùng chồng lấp
            main_contour = max(overlap_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            overlap_region = (x, y, w, h)
            
            # Sử dụng multiband blending (đã được tối ưu hóa)
            # Chuẩn bị mask cho multiband blending
            blend_mask = self.create_smoothing_mask(panorama, warped2, overlap_region)
            blend_mask_255 = (blend_mask * 255).astype(np.uint8)
            
            # Thực hiện multiband blending
            panorama = self.apply_multiband_blending(panorama, warped2, blend_mask_255)
        else:
            # Không có vùng chồng lấp, ghép trực tiếp
            # Lấy phần không đen từ warped2
            mask = (warped2 > 0).astype(np.uint8)
            # Chỉ cập nhật các pixel không đen từ warped2
            panorama = panorama * (1 - mask) + warped2 * mask
        
        # 9. Crop tự động để bỏ viền đen
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Tìm contour lớn nhất (nội dung ảnh)
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            final = panorama[y:y+h, x:x+w]
        else:
            final = panorama
        
        # 10. Nâng cao chất lượng ảnh
        final = self.enhance_image(final)
        
        # 11. Khôi phục kích thước gốc nếu cần
        if scale1 != 1.0 or scale2 != 1.0:
            # Tính tỷ lệ khôi phục
            restore_scale = 1.0 / scale1  # Sử dụng tỷ lệ của ảnh 1 làm chuẩn
            new_size = (int(final.shape[1] * restore_scale), int(final.shape[0] * restore_scale))
            final = cv2.resize(final, new_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"Đã khôi phục kích thước panorama về {new_size[0]}x{new_size[1]}")
        
        return final, matches_image
    
    def stitch_multiple_images(self, images, output_path=None):
        """
        Ghép nhiều ảnh thành một panorama
        
        Args:
            images: Danh sách các ảnh đầu vào
            output_path: Đường dẫn để lưu ảnh kết quả
            
        Returns:
            result: Ảnh panorama cuối cùng
        """
        if len(images) < 2:
            raise ValueError("Cần ít nhất 2 ảnh để ghép panorama")
        
        # Bắt đầu với ảnh đầu tiên
        result = images[0]
        
        # Ghép lần lượt các ảnh còn lại
        for i, img in enumerate(images[1:], 1):
            print(f"\nĐang ghép ảnh {i}/{len(images)-1}...")
            result, _ = self.stitch_images(result, img)
        
        # Lưu ảnh kết quả nếu được chỉ định
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Đã lưu panorama tại: {output_path}")
        
        return result

def select_images():
    """
    Cho phép người dùng chọn nhiều ảnh từ hệ thống
    
    Returns:
        list: Danh sách đường dẫn các ảnh đã chọn
    """
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ gốc Tkinter
    
    file_paths = filedialog.askopenfilenames(
        title="Chọn các ảnh để ghép panorama",
        filetypes=[
            ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Tất cả file", "*.*")
        ]
    )
    
    return list(file_paths)

if __name__ == "__main__":
    print("Chương trình Ghép Ảnh Panorama Tự Động")
    print("======================================")
    
    # Tạo thư mục outputs nếu chưa tồn tại
    os.makedirs("outputs", exist_ok=True)
    
    # Hiện cửa sổ chọn ảnh
    img_paths = select_images()
    
    if not img_paths or len(img_paths) < 2:
        print("Cần chọn ít nhất 2 ảnh để tạo panorama. Kết thúc chương trình.")
        sys.exit(0)
        
    print(f"Đã chọn {len(img_paths)} ảnh.")
    
    # Đọc các ảnh đã chọn
    print("\nĐọc các ảnh đầu vào...")
    images = []
    for path in img_paths:
        print(f"Đang đọc: {path}")
        img = cv2.imread(path)
        if img is None:
            print(f"Cảnh báo: Không thể đọc ảnh '{path}'. Bỏ qua.")
            continue
        images.append(img)
    
    # Kiểm tra số lượng ảnh hợp lệ
    if len(images) < 2:
        print(f"Lỗi: Cần ít nhất 2 ảnh để ghép panorama. Chỉ đọc được {len(images)} ảnh.")
        sys.exit(1)
    
    # Hiển thị các ảnh đầu vào
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.title(f"Ảnh {i+1}: {os.path.basename(img_paths[i])}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    # Tạo stitcher với các cài đặt tối ưu
    stitcher = PanoramaStitcher(max_features=10000, ratio_thresh=0.5, reproj_thresh=4.0)
    
    # Tạo tên file dựa trên thời gian
    output_path = os.path.join(output_dir, f"result_task2.jpg")    
    # Ghép ảnh và hiển thị kết quả
    try:
        print("\nĐang xử lý... Vui lòng đợi...")
        panorama = stitcher.stitch_multiple_images(images, output_path=output_path)
        
        # Hiển thị kết quả cuối cùng
        plt.figure(figsize=(16, 10))
        plt.title("Panorama Hoàn Chỉnh")
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        print(f"\nQuy trình ghép ảnh hoàn tất!")
        
    except Exception as e:
        print(f"Lỗi trong quá trình ghép ảnh: {str(e)}")
        import traceback
        traceback.print_exc() 