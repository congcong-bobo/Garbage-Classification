#主要功能
#实时显示分类结果和置信度
#支持按键保存当前帧和推理结果
#结构化存储反馈数据（图像 + 元数据）


import tensorflow as tf
import numpy as np
import cv2
from camera_capture import CameraCapture
import threading
import json
import os
from datetime import datetime

class RealTimeInference:
    def __init__(self, model_path, camera_id=0):
        self.model = tf.keras.models.load_model(model_path)
        self.camera = CameraCapture(camera_id)
        self.running = False
        self.result_queue = queue.Queue()
        self.feedback_data = []
        self.inference_thread = None
        self.display_thread = None
        
    def start(self):
        """启动实时推理系统"""
        self.camera.start()
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        print("实时推理系统已启动")
        
    def _preprocess_frame(self, frame):
        """预处理图像帧用于模型推理"""
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0  # 归一化
        img = np.expand_dims(img, axis=0)
        return img
        
    def _inference_loop(self):
        """推理循环，运行在独立线程中"""
        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue
                
            # 预处理图像
            input_img = self._preprocess_frame(frame)
            
            # 模型推理
            predictions = self.model.predict(input_img, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            # 获取类别名称（假设我们有这些类别）
            classes = ['可回收物', '有害垃圾', '厨余垃圾', '其他垃圾']
            predicted_class = classes[class_idx]
            
            # 将结果存入队列
            result = {
                'timestamp': datetime.now().isoformat(),
                'class': predicted_class,
                'confidence': confidence,
                'distribution': {cls: float(p) for cls, p in zip(classes, predictions)}
            }
            self.result_queue.put(result)
            
    def _display_loop(self):
        """显示循环，运行在独立线程中"""
        while self.running:
            frame = self.camera.last_frame
            if frame is None:
                continue
                
            # 获取最新的推理结果
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                result = None
                
            # 在图像上绘制结果
            display_frame = frame.copy()
            if result:
                cv2.putText(display_frame, 
                           f"分类: {result['class']} ({result['confidence']:.2%})",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (0, 255, 0),
                           2)
                
            # 显示图像
            cv2.imshow('实时垃圾分类', display_frame)
            
            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 'q' 键退出
                self.stop()
            elif key == ord('s'):  # 按 's' 键保存当前帧和结果
                self.save_feedback(frame, result)
                
    def stop(self):
        """停止实时推理系统"""
        self.running = False
        self.camera.stop()
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        if self.display_thread:
            cv2.destroyAllWindows()
            self.display_thread.join(timeout=2.0)
        print("实时推理系统已停止")
        
    def save_feedback(self, frame=None, result=None):
        """保存反馈数据用于模型优化"""
        if frame is None:
            frame = self.camera.last_frame
            
        if result is None:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                print("没有可用的推理结果")
                return False
                
        # 保存图像
        timestamp = result['timestamp'].replace(':', '-').replace('.', '_')
        img_path = f"feedback_data/images/{timestamp}.jpg"
        self.camera.save_frame(frame, img_path)
        
        # 保存元数据
        metadata = {
            'image_path': img_path,
            'prediction': result,
            'user_feedback': None,  # 待用户标注
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = f"feedback_data/metadata/{timestamp}.json"
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        self.feedback_data.append(metadata)
        print(f"反馈数据已保存: {img_path}")
        return True
        
    def get_latest_result(self):
        """获取最新的推理结果"""
        try:
            # 清空队列，只保留最新结果
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
            return result
        except (queue.Empty, UnboundLocalError):
            return None
            
    def get_feedback_data(self):
        """获取所有反馈数据"""
        return self.feedback_data