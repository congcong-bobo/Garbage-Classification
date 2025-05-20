# generate_data.py
import numpy as np
import pandas as pd
import os
from config import CLASSES  # 导入统一的类别定义

# 设置随机种子，确保结果可复现
np.random.seed(42)

def generate_synthetic_odor_data(num_samples_per_class=100):
    """生成模拟的气味传感器数据集"""
    # 假设我们有10个气体传感器
    num_sensors = 10
    
    # 为每个类别创建特征分布（均值和标准差）
    class_means = np.array([
        # harmful_waste: 可能含有特殊化学成分
        [3, 2, 4, 5, 6, 3, 2, 1, 7, 4],
        # kitchen_garbage: 高湿度、易挥发有机物
        [8, 7, 9, 8, 4, 3, 5, 6, 9, 7],
        # other: 成分复杂，响应波动大
        [4, 3, 5, 4, 3, 2, 4, 3, 5, 4],
        # recyclable_garbage: 低响应，可能有特定物质残留
        [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    ])
    
    # 每个类别的标准差
    class_stds = np.array([
        [0.8, 0.6, 1.0, 1.2, 1.5, 0.8, 0.6, 0.5, 1.8, 1.0],
        [1.5, 1.2, 2.0, 1.5, 1.0, 0.8, 1.2, 1.5, 2.0, 1.5],
        [1.0, 0.8, 1.2, 1.0, 0.8, 0.6, 1.0, 0.8, 1.2, 1.0],
        [0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3]
    ])
    
    # 生成数据
    X = []
    y = []
    
    for class_idx, (means, stds) in enumerate(zip(class_means, class_stds)):
        # 为每个类别生成样本
        samples = np.random.normal(means, stds, size=(num_samples_per_class, num_sensors))
        # 确保特征非负（传感器响应通常为正值）
        samples = np.maximum(0, samples)
        X.append(samples)
        y.extend([class_idx] * num_samples_per_class)
    
    # 合并所有类别数据
    X = np.vstack(X)
    y = np.array(y)
    
    # 创建DataFrame
    columns = [f"传感器{i+1}" for i in range(num_sensors)]
    df = pd.DataFrame(X, columns=columns)
    df["类别"] = [CLASSES[i] for i in y]  # 使用统一的类别名称
    
    return df

if __name__ == "__main__":
    odor_data = generate_synthetic_odor_data(num_samples_per_class=200)
    
    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)
    
    odor_data.to_csv("data/garbage_odor_data.csv", index=False)
    print("数据集已生成并保存至 data/garbage_odor_data.csv")