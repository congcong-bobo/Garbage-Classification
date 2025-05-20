#主要功能
#收集用户标注的反馈数据
#自动整理训练数据目录结构
#微调预训练模型，持续改进分类效果


import os
import json
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class DataFeedbackLoop:
    def __init__(self, base_model_path, feedback_dir="feedback_data"):
        self.base_model_path = base_model_path
        self.feedback_dir = feedback_dir
        self.metadata_dir = os.path.join(feedback_dir, "metadata")
        self.images_dir = os.path.join(feedback_dir, "images")
        self.processed_dir = os.path.join(feedback_dir, "processed")
        self.training_dir = os.path.join(feedback_dir, "training_data")
        
    def collect_annotated_data(self):
        """收集已标注的反馈数据"""
        annotated_data = []
        
        if not os.path.exists(self.metadata_dir):
            print(f"元数据目录不存在: {self.metadata_dir}")
            return annotated_data
            
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith('.json'):
                metadata_path = os.path.join(self.metadata_dir, filename)
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        
                    # 检查是否有用户反馈
                    if metadata.get('user_feedback') is not None:
                        annotated_data.append(metadata)
                except Exception as e:
                    print(f"读取元数据文件 {metadata_path} 时出错: {e}")
                    
        return annotated_data
        
    def prepare_training_data(self, annotated_data):
        """准备训练数据"""
        if not annotated_data:
            print("没有可用的标注数据")
            return False
            
        # 创建训练目录结构
        classes = ['可回收物', '有害垃圾', '厨余垃圾', '其他垃圾']
        for cls in classes:
            os.makedirs(os.path.join(self.training_dir, 'train', cls), exist_ok=True)
            os.makedirs(os.path.join(self.training_dir, 'validation', cls), exist_ok=True)
            
        # 划分训练集和验证集
        train_data, val_data = train_test_split(annotated_data, test_size=0.2, random_state=42)
        
        # 复制图像到训练目录
        for data in train_data:
            src_img = data['image_path']
            if not os.path.exists(src_img):
                continue
                
            cls = data['user_feedback']
            dst_img = os.path.join(self.training_dir, 'train', cls, os.path.basename(src_img))
            shutil.copy(src_img, dst_img)
            
        for data in val_data:
            src_img = data['image_path']
            if not os.path.exists(src_img):
                continue
                
            cls = data['user_feedback']
            dst_img = os.path.join(self.training_dir, 'validation', cls, os.path.basename(src_img))
            shutil.copy(src_img, dst_img)
            
        print(f"训练数据准备完成: {len(train_data)} 训练样本, {len(val_data)} 验证样本")
        return True
        
    def retrain_model(self, epochs=5):
        """基于反馈数据重新训练模型"""
        # 收集标注数据
        annotated_data = self.collect_annotated_data()
        if not self.prepare_training_data(annotated_data):
            return None
            
        # 加载基础模型
        base_model = load_model(self.base_model_path)
        
        # 解冻最后几层进行微调
        for layer in base_model.layers[-20:]:
            layer.trainable = True
            
        # 编译模型
        base_model.compile(
            optimizer=Adam(lr=0.0001),  # 较小的学习率用于微调
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 数据生成器
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # 加载数据
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.training_dir, 'train'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            os.path.join(self.training_dir, 'validation'),
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        
        # 训练模型
        history = base_model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epochs
        )
        
        # 保存微调后的模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_path = f"models/improved_model_{timestamp}.h5"
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        base_model.save(new_model_path)
        
        # 绘制训练历史
        self._plot_training_history(history)
        
        print(f"模型重新训练完成并保存至: {new_model_path}")
        return new_model_path
        
    def _plot_training_history(self, history):
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('模型准确率')
        plt.ylabel('准确率')
        plt.xlabel('训练轮次')
        plt.legend(['训练', '验证'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('模型损失')
        plt.ylabel('损失')
        plt.xlabel('训练轮次')
        plt.legend(['训练', '验证'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")