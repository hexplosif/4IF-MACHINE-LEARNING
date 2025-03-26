import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Đường dẫn thư mục dữ liệu
DATA_PATH = '/Users/sonngo/Documents/Workspaces/insa/4if/ML/4IF-MACHINE-LEARNING/data/classification'

def load_data():
    """
    Tải dữ liệu train và test từ thư mục
    
    Returns:
        tuple: Dataframe train và test
    """
    train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
    test_df = pd.read_csv(f'{DATA_PATH}/test.csv')
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Tiền xử lý dữ liệu cho mô hình
    
    Args:
        train_df (pd.DataFrame): Dữ liệu huấn luyện
        test_df (pd.DataFrame): Dữ liệu kiểm tra
    
    Returns:
        tuple: X_train, X_test, y_train, test_index
    """
    # Chọn các đặc trưng
    features = ['date', 'hour', 'bc_price', 'bc_demand', 
                'ab_price', 'ab_demand', 'transfer']
    
    # Tách đặc trưng và nhãn cho tập train
    X_train = train_df[features]
    y_train = train_df['bc_price_evo']
    
    # Đặc trưng cho tập test
    X_test = test_df[features]
    test_index = test_df.iloc[:, 0]  # Lấy index ban đầu
    
    return X_train, X_test, y_train, test_index

def train_and_predict_model(X_train, X_test, y_train):
    """
    Huấn luyện mô hình và dự đoán
    
    Args:
        X_train (pd.DataFrame): Đặc trưng huấn luyện
        X_test (pd.DataFrame): Đặc trưng kiểm tra
        y_train (pd.Series): Nhãn huấn luyện
    
    Returns:
        np.ndarray: Nhãn dự đoán cho tập test
    """
    # Tạo pipeline với RandomForest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)
    
    # Dự đoán nhãn cho tập test
    y_pred = pipeline.predict(X_test)
    
    return y_pred

def save_predictions(test_index, predictions):
    """
    Lưu các dự đoán vào file CSV
    
    Args:
        test_index (pd.Series): Index gốc của tập test
        predictions (np.ndarray): Các nhãn dự đoán
    """
    # Tạo DataFrame kết quả
    results_df = pd.DataFrame({
        'id': test_index,
        'bc_price_evo': predictions
    })
    
    # Lưu file CSV
    results_df.to_csv(f'{DATA_PATH}/submission/chatGPT_code.csv', index=False)
    
    print("Đã lưu file dự đoán: test_predictions.csv")

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình
    """
    # Tải dữ liệu
    train_df, test_df = load_data()
    
    # Tiền xử lý dữ liệu
    X_train, X_test, y_train, test_index = preprocess_data(train_df, test_df)
    
    # Huấn luyện và dự đoán
    predictions = train_and_predict_model(X_train, X_test, y_train)
    
    # Lưu kết quả
    save_predictions(test_index, predictions)
    
    # In báo cáo đánh giá (tuỳ chọn)
    print("\nBáo cáo chi tiết:")
    print("Tổng số mẫu test:", len(predictions))
    unique, counts = np.unique(predictions, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"Nhãn {value}: {count} mẫu")

# Chạy chương trình chính
if __name__ == "__main__":
    main()