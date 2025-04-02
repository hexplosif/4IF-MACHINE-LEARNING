import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Khởi tạo các tham số
DATA_PATH = '/Users/sonngo/Documents/Workspaces/insa/4if/ML/4IF-MACHINE-LEARNING/data/classification/'
START_DATE = pd.to_datetime('2015-05-15')
TOTAL_DAYS = 942  # Từ 15/5/2015 đến 13/12/2017

def preprocess_data(df):
    """Tiền xử lý dữ liệu và tạo features"""
    # Chuyển đổi ngày thực tế
    df['date_actual'] = START_DATE + pd.to_timedelta(df['date'] * TOTAL_DAYS, unit='D')
    
    # Trích xuất các thành phần thời gian
    df['year'] = df['date_actual'].dt.year
    df['month'] = df['date_actual'].dt.month
    df['day_of_week'] = df['date_actual'].dt.dayofweek
    
    # Chuyển đổi giờ về dạng nguyên bản (0-47)
    df['hour_original'] = (df['hour'] * 47).round().astype(int)
    
    # Tạo features tuần hoàn
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_original'] / 48)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_original'] / 48)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def main():
    # Đọc dữ liệu
    train = pd.read_csv(f'{DATA_PATH}train.csv', index_col=0)
    test = pd.read_csv(f'{DATA_PATH}test.csv', index_col=0)
    
    # Tiền xử lý
    train = preprocess_data(train)
    test = preprocess_data(test)
    
    # Danh sách features
    features = [
        'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer',
        'year', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'hour_sin', 'hour_cos'
    ]
    
    # Chuẩn bị dữ liệu training
    train_sorted = train.sort_values(['date_actual', 'hour_original'])
    X_train = train_sorted[features]
    y_train = train_sorted['bc_price_evo']
    
    # Mã hóa nhãn
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Chia tập validation theo thời gian
    split_idx = int(0.8 * len(X_train))
    X_train_split, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_train_split, y_val = y_train_encoded[:split_idx], y_train_encoded[split_idx:]
    
    # Huấn luyện mô hình
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.05
    )
    model.fit(X_train_split, y_train_split)
    
    # Dự đoán trên tập test
    test_sorted = test.sort_values(['date_actual', 'hour_original'])
    X_test = test_sorted[features]
    
    test_pred_encoded = model.predict(X_test)
    test_pred = le.inverse_transform(test_pred_encoded)
    
    # Tạo file submission
    submission = test_sorted[[]].copy()
    submission['bc_price_evo'] = test_pred
    submission = submission.sort_index()[['bc_price_evo']]  # Khôi phục thứ tự gốc
    
    # Lưu kết quả
    submission.to_csv(f'{DATA_PATH}Deepseek_code.csv')
    print("Saved predictions to predictions.csv")

if __name__ == "__main__":
    main()