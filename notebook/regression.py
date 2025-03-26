# 导入所需的库
import numpy as np
import pandas as pd
import os

# 列出输入目录下的文件（验证数据路径是否正确）
for dirname, _, filenames in os.walk("/kaggle/input/insa-ml-2025-regression"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 加载数据
train_df = pd.read_csv("/kaggle/input/insa-ml-2025-regression/train.csv")
test_df = pd.read_csv("/kaggle/input/insa-ml-2025-regression/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/insa-ml-2025-regression/sample_submission.csv"
)

# 查看数据结构
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# 将训练集和测试集合并，便于统一处理分类变量（注意保留目标变量信息）
train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["co2"] = np.nan  # 占位，方便合并

data = pd.concat([train_df, test_df], sort=False)

# 简单预处理：
# 1. 去除id字段（后续不会用于模型训练）；
# 2. 对文本类变量进行one-hot编码；
# 3. 数值型变量缺失值填充（这里用中位数填充）

# 去除id字段并记录
data_index = data["id"]
data = data.drop("id", axis=1)

# 处理缺失值（此处仅为示例，可以根据实际数据情况改进）
for col in data.columns:
    if data[col].dtype in ["float64", "int64"]:
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].fillna("missing")

# 将分类变量进行one-hot编码（这里对object类型变量进行编码）
data = pd.get_dummies(data, drop_first=True)

# 将数据分离为训练集和测试集
train_data = data[data["is_train"] == 1].drop("is_train", axis=1)
test_data = data[data["is_train"] == 0].drop(
    ["is_train", "co2"], axis=1
)  # 测试集没有co2

# 提取训练数据中的目标变量和特征
y_train = train_data["co2"]
X_train = train_data.drop("co2", axis=1)

print("Processed X_train shape:", X_train.shape)
print("Processed test_data shape:", test_data.shape)

# 使用随机森林模型进行训练（你也可以使用其他模型，如LightGBM、XGBoost、Keras等）
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# 设置随机种子，便于结果复现
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 交叉验证评估模型性能
cv_scores = cross_val_score(
    rf, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
)
print(
    "Cross-validation MAE: {:.2f} ± {:.2f}".format(-cv_scores.mean(), cv_scores.std())
)

# 在整个训练集上训练模型
rf.fit(X_train, y_train)

# 对测试集进行预测
test_preds = rf.predict(test_data)

# 构造提交文件，要求格式为：id,co2
submission = pd.DataFrame(
    {
        "id": sample_submission["id"],  # 保持与sample_submission中的id顺序一致
        "co2": test_preds.astype(int),  # 可根据需要转换为整数
    }
)

# 保存提交文件
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
