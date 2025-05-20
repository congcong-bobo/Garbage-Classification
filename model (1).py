# ainose_classifier/model.py
import os
import joblib
import numpy as np
from config import CLASSES, CLASS_MAPPING  # 导入统一的类别定义和类别映射字典

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型和标准化器的绝对路径
model_path = os.path.join(current_dir, 'models', 'odor_classifier_rf.pkl')
scaler_path = os.path.join(current_dir, 'models', 'odor_scaler.pkl')

# 加载模型和标准化器
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def odor_classify(sensor_data):
    """预测嗅觉数据类别及置信度"""
    # 标准化数据
    scaled_data = scaler.transform([sensor_data])
    
    # 预测类别
    predicted_class = rf_model.predict(scaled_data)[0]
    
    # 检查预测结果是否在 CLASSES 列表中
    if predicted_class not in CLASSES:
        # 使用类别映射字典处理未知类别
        predicted_class = CLASS_MAPPING.get(predicted_class, "未知类别")
    
    # 获取预测概率
    confidence = np.max(rf_model.predict_proba(scaled_data))
    
    print(f"预测类别: {predicted_class}, 置信度: {confidence:.2%}")
    return predicted_class, confidence