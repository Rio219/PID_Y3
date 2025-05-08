import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os, zipfile, kagglehub

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def download_dataset():
    print("Loading FER2013…")
    result = kagglehub.dataset_download("msambare/fer2013")
    # Nếu result là file ZIP, giải nén:
    if result.endswith('.zip'):
        target = 'fer2013'
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(result, 'r') as z:
            z.extractall(target)
        return target
    else:
        return result

def load_images(dataset_path, subset='train'):
    """
    Đọc ảnh grayscale từ thư mục tương ứng
    
    Parameters:
    -----------
    dataset_path: đường dẫn đến bộ dữ liệu
    subset: 'train' hoặc 'test'
    
    Returns:
    --------
    images: danh sách ảnh
    labels: danh sách nhãn tương ứng
    label_names: tên các lớp
    """
    images = []
    labels = []
    subset_path = os.path.join(dataset_path, subset)
    label_names = os.listdir(subset_path)
    
    for label_idx, emotion in enumerate(label_names):
        emotion_path = os.path.join(subset_path, emotion)
        if os.path.isdir(emotion_path):
            print(f"Đang đọc ảnh từ thư mục {emotion}...")
            for img_file in os.listdir(emotion_path):
                if img_file.endswith(('.jpg', '.png')):
                    img_path = os.path.join(emotion_path, img_file)
                    img = io.imread(img_path, as_gray=True)
                    img = cv2.equalizeHist(img)         # cân bằng histogram
                    images.append(img)
                    labels.append(label_idx)
    
    return np.array(images), np.array(labels), label_names

def extract_lbph_features(images, P=8, R=1, method='uniform'):
    """
    Trích xuất đặc trưng LBPH từ ảnh
    
    Parameters:
    -----------
    images: danh sách ảnh
    P: số điểm lân cận (mặc định: 8)
    R: bán kính (mặc định: 1)
    method: phương pháp LBP ('uniform', 'default', 'ror', 'var')
    
    Returns:
    --------
    features: ma trận đặc trưng LBPH
    """
    n_points = P * R
    # Tính số bins dựa trên method và n_points
    if method == 'uniform':
        n_bins = n_points + 2
    else:
        n_bins = 2**n_points
    
    print(f"Trích xuất đặc trưng LBPH với P={P}, R={R}, method={method}, n_bins={n_bins}...")
    
    features = []
    for img in images:
        # Trích xuất LBP
        lbp = local_binary_pattern(img, n_points, R, method)
        # Tính histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        # Chuẩn hóa histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # L1 norm
        features.append(hist)
    
    return np.array(features)

def train_and_evaluate(X_train, y_train, X_test, y_test, label_names):
    """
    Huấn luyện Random Forest và đánh giá mô hình
    
    Parameters:
    -----------
    X_train, y_train: dữ liệu huấn luyện
    X_test, y_test: dữ liệu kiểm tra
    label_names: tên các lớp
    n_estimators: số cây quyết định trong rừng
    max_depth: độ sâu tối đa của cây
    
    Returns:
    --------
    model: mô hình đã huấn luyện
    accuracy: độ chính xác
    report: báo cáo phân loại
    """

    # Chuẩn hóa đặc trưng
    print("Chuẩn hóa đặc trưng...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print("Tham số tốt nhất:", grid.best_params_)
    model = grid.best_estimator_

    # Dự đoán và đánh giá
    print("Đánh giá mô hình...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)

    # Hiển thị confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return model, accuracy, report, y_pred


def visualize_predictions(test_images, y_test, y_pred, label_names, n_samples=16):
    """
    Hiển thị một số dự đoán ngẫu nhiên theo dạng lưới
    
    Parameters:
    -----------
    test_images : danh sách ảnh kiểm tra
    y_test      : nhãn thật
    y_pred      : nhãn dự đoán
    label_names : danh sách tên lớp
    n_samples   : số lượng ảnh muốn hiển thị
    """
    n_samples = min(n_samples, len(test_images))
    indices = np.random.choice(len(test_images), n_samples, replace=False)

    rows = 4
    cols = 4
    plt.figure(figsize=(16, 12))

    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(test_images[idx], cmap='gray')
        actual = label_names[y_test[idx]]
        predicted = label_names[y_pred[idx]]
        plt.title(f'Thực tế: {actual}\nDự đoán: {predicted}', fontsize=10)
        plt.axis('off')

    plt.tight_layout()

    # Lưu ảnh vào thư mục outputs cùng cấp với file code
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'result_task4.jpg')
    plt.savefig(output_path)
    print(f"Đã lưu ảnh dự đoán tại: {output_path}")

    plt.show()
def main():
    # Tải dữ liệu
    dataset_path = download_dataset()
    
    # Đọc dữ liệu huấn luyện
    train_images, train_labels, label_names = load_images(dataset_path, 'train')
    print(f"Đã tải {len(train_images)} ảnh huấn luyện từ {len(label_names)} lớp: {label_names}")
    
    # Đọc dữ liệu kiểm tra
    test_images, test_labels, _ = load_images(dataset_path, 'test')
    print(f"Đã tải {len(test_images)} ảnh kiểm tra")
    
    # Trích xuất đặc trưng LBPH
    # P: số điểm lân cận, R: bán kính, method: phương pháp LBP
    P, R = 16, 2
    train_features = extract_lbph_features(train_images, P, R, method='uniform')
    test_features = extract_lbph_features(test_images, P, R, method='uniform')

    # Huấn luyện và đánh giá mô hình
    model, accuracy, report, y_pred = train_and_evaluate(
        train_features, train_labels, 
        test_features, test_labels,
        label_names
)

    
    # In kết quả
    print(f"\nĐộ chính xác: {accuracy*100:.2f}%")
    print("\nBáo cáo phân loại chi tiết:")
    print(report)
    
    # Hiển thị dự đoán
    visualize_predictions(test_images, test_labels, y_pred, label_names)

if __name__ == "__main__":
    main() 