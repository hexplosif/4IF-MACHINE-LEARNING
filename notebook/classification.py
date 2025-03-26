# 导入常用库
import numpy as np
import pandas as pd
import os

# 输出数据文件路径
for dirname, _, filenames in os.walk("/kaggle/input/insa-ml-2025-classification"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 数据路径
data_path = "/kaggle/input/insa-ml-2025-classification/"

# 加载训练集和测试集数据
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# 将目标变量进行编码：UP -> 1, DOWN -> 0
train_df["bc_price_evo"] = train_df["bc_price_evo"].map({"UP": 1, "DOWN": 0})

# 选择特征（去掉 id 和目标变量）
features = [
    "date",
    "hour",
    "bc_price",
    "bc_demand",
    "ab_price",
    "ab_demand",
    "transfer",
]
X_train = train_df[features]
y_train = train_df["bc_price_evo"]
X_test = test_df[features]

# 导入需要的模型
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 初始化各个模型，适当调大了树的个数和深度
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1,
)
lgb = LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1
)

# 建立集成模型（软投票），对每个基模型预测概率求平均
voting_clf = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("lgb", lgb)], voting="soft", n_jobs=-1
)

# 使用交叉验证评估集成模型（可选步骤，利用充足内存）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    voting_clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
)
print("Cross-validation scores:", cv_scores)
print("Mean CV score: {:.4f}".format(cv_scores.mean()))

# 在整个训练集上训练模型
voting_clf.fit(X_train, y_train)

# 在训练集上评估模型表现
train_preds = voting_clf.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
print("Ensemble training accuracy:", train_acc)

# 在测试集上进行预测
test_preds = voting_clf.predict(X_test)
# 将数值型预测转换为对应的标签：1->UP, 0->DOWN
test_preds_labels = np.where(test_preds == 1, "UP", "DOWN")

# 生成提交文件，格式：id,bc_price_evo
submission = pd.DataFrame({"id": test_df["id"], "bc_price_evo": test_preds_labels})

# 查看前几行预测结果
print(submission.head())

# 保存提交文件
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
