import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from config import CLASSES  # 导入统一的类别定义

# 读取数据集
odor_data = pd.read_csv("data/garbage_odor_data.csv")

# 过滤数据，只保留目标类别的数据
odor_data = odor_data[odor_data['类别'].isin(CLASSES)]

# 分离特征和标签
X = odor_data.drop("类别", axis=1)
y = odor_data["类别"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,  # 决策树数量
    max_depth=10,      # 树的最大深度
    random_state=42,
    n_jobs=-1          # 使用所有CPU核心
)

# 拟合模型
rf_model.fit(X_train, y_train)

# 在测试集上预测
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# 评估模型
print("分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=rf_model.classes_, 
            yticklabels=rf_model.classes_)
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("气味分类混淆矩阵")
plt.show()

# 特征重要性可视化
plt.figure(figsize=(12, 6))
feature_importance = rf_model.feature_importances_
feature_names = X.columns
plt.bar(feature_names, feature_importance)
plt.xlabel("传感器")
plt.ylabel("特征重要性")
plt.title("各传感器对分类的重要性")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 保存模型和标准化器
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/odor_classifier_rf.pkl")
joblib.dump(scaler, "models/odor_scaler.pkl")
print("模型和标准化器已保存至 models 目录")