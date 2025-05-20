#主要功能
#使用 OpenCV 从摄像头实时获取图像
#独立线程处理图像采集，不阻塞主线程
#图像队列管理，防止内存溢出
import cv2
import threading
import queue
from datetime import datetime
import os

class CameraCapture:
    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)  # 限制队列大小防止内存溢出
        self.thread = None
        self.last_frame = None
        
    def start(self):
        """启动摄像头并开始采集图像"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头 {self.camera_id}")
            
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"摄像头 {self.camera_id} 已启动")
        
    def _capture_loop(self):
        """摄像头采集循环，运行在独立线程中"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取帧，退出采集循环")
                break
                
            # 如果队列已满，丢弃最旧的帧
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
                
            self.frame_queue.put(frame)
            self.last_frame = frame  # 保存最新帧用于快速预览
            
        self.cap.release()
        
    def get_frame(self, timeout=1.0):
        """获取一帧图像，带超时处理"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return self.last_frame  # 如果队列为空，返回最后一帧
            
    def stop(self):
        """停止摄像头采集"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("摄像头已停止")

    def save_frame(self, frame=None, path=None):
        """保存当前帧到指定路径"""
        if frame is None:
            frame = self.last_frame
            
        if frame is None:
            print("没有可用的帧进行保存")
            return False
            
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"captured_frames/frame_{timestamp}.jpg"
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return cv2.imwrite(path, frame)