import os
import cv2
import numpy as np
import yaml
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QInputDialog, 
                            QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from scipy.spatial.transform import Rotation as R

class HandEyeCalibrationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_config()
        self.init_camera()
        self.init_data_storage()  # 初始化时自动生成符合格式的保存目录
        
    def init_ui(self):
        self.setWindowTitle("手眼标定数据采集")
        self.setGeometry(100, 100, 1000, 800)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 摄像头显示区域
        self.camera_label = QLabel("摄像头画面")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.camera_label)
        
        # 状态显示
        self.status_label = QLabel("就绪 - 按 's' 采集数据，按 'q' 退出")
        main_layout.addWidget(self.status_label)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始采集")
        self.start_btn.clicked.connect(self.start_capture)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止采集")
        self.stop_btn.clicked.connect(self.stop_capture)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.save_dir_btn = QPushButton("选择保存目录")
        self.save_dir_btn.clicked.connect(self.choose_save_dir)
        btn_layout.addWidget(self.save_dir_btn)
        
        main_layout.addLayout(btn_layout)
        
        # 设置主窗口部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 变量初始化
        self.capture_active = False
        self.capture_count = 0
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_hand_data")  # 基础目录
        self.save_dir = ""  # 会在init_data_storage中自动生成
        self.poses_file = ""
        
        # 创建定时器更新摄像头画面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def load_config(self):
        """加载标定板配置参数"""
        try:
            with open("config.yaml", 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            self.XX = data.get("checkerboard_args").get("XX")  # 标定板长度方向角点个数
            self.YY = data.get("checkerboard_args").get("YY")  # 标定板宽度方向角点个数
            self.L = data.get("checkerboard_args").get("L")    # 标定板格子长度(米)
            
            self.W = data.get("W")
            self.H = data.get("H")
            # 设置亚像素角点检测参数
            self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
            
        except Exception as e:
            QMessageBox.critical(self, "配置错误", f"加载配置文件失败: {str(e)}")
            self.XX, self.YY, self.L = 9, 6, 0.02  # 默认值
    
    def init_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(0)  # 默认摄像头
        # 设置分辨率（可选，根据需要调整）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "摄像头错误", "无法打开摄像头，请检查设备连接")
            self.close()
        
        # 验证实际分辨率
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头实际分辨率: {actual_width}x{actual_height}")
    
    def get_next_save_dir(self):
        """自动生成下一个保存目录（格式：dataYYYYMMDDXX）"""
        # 获取当前日期（YYYYMMDD格式）
        today = datetime.now().strftime("%Y%m%d")
        base_name = f"data{today}"
        
        # 确保基础目录存在
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # 查找当前日期下已存在的组号
        existing_dirs = []
        for dir_name in os.listdir(self.base_dir):
            if dir_name.startswith(base_name) and len(dir_name) == len(base_name) + 2:
                # 提取末尾的两位数字
                suffix = dir_name[-2:]
                if suffix.isdigit():
                    existing_dirs.append(int(suffix))
        
        # 确定下一个组号（两位数字，从01开始）
        if existing_dirs:
            next_num = max(existing_dirs) + 1
        else:
            next_num = 1
        
        # 格式化为两位数字（01, 02, ..., 99）
        next_suffix = f"{next_num:02d}"
        new_dir = os.path.join(self.base_dir, f"{base_name}{next_suffix}")
        
        return new_dir
    
    def init_data_storage(self):
        """初始化数据存储目录（自动生成格式：dataYYYYMMDDXX）"""
        # 自动获取下一个保存目录
        self.save_dir = self.get_next_save_dir()
        
        # 创建目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 初始化位姿文件路径
        self.poses_file = os.path.join(self.save_dir, "poses.txt")
        
        # 清空已有poses.txt文件
        with open(self.poses_file, 'w') as f:
            pass
        
        self.status_label.setText(f"保存目录已创建: {self.save_dir}")
    
    def choose_save_dir(self):
        """选择保存目录（保留手动选择功能，可选）"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.base_dir)
        if dir_path:
            self.save_dir = dir_path
            self.poses_file = os.path.join(self.save_dir, "poses.txt")
            # 如果目录不存在则创建
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            # 清空或创建poses.txt
            with open(self.poses_file, 'w') as f:
                pass
            self.status_label.setText(f"保存目录已设置: {self.save_dir}")
    
    def start_capture(self):
        """开始采集"""
        self.capture_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(f"采集已开始 - 保存至: {os.path.basename(self.save_dir)} - 按 's' 采集数据，按 'q' 退出")
        self.timer.start(30)  # 约33fps
    
    def stop_capture(self):
        """停止采集"""
        self.capture_active = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"采集已停止 - 共采集 {self.capture_count} 组数据 - 保存至: {os.path.basename(self.save_dir)}")
        self.timer.stop()
    
    def update_frame(self):
        """更新摄像头画面并检测棋盘格"""
        ret, frame = self.cap.read()
        if ret:
            # 检测棋盘格
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(gray, (self.XX, self.YY), None)
            
            # 如果检测到角点，绘制角点
            if ret_corners:
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(frame, (self.XX, self.YY), corners2, ret_corners)
                self.status_label.setText(f"检测到棋盘格 - 保存至: {os.path.basename(self.save_dir)} - 按 's' 采集第 {self.capture_count + 1} 组数据")
            else:
                self.status_label.setText(f"未检测到棋盘格 - 保存至: {os.path.basename(self.save_dir)} - 请调整摄像头位置")
            
            # 转换为Qt格式并显示
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, w * c, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio))
    
    def keyPressEvent(self, event):
        """处理键盘事件"""
        if event.key() == Qt.Key_S and self.capture_active:
            self.capture_data()
        elif event.key() == Qt.Key_Q:
            self.close()
    
    def capture_data(self):
        """采集数据"""
        # 获取当前帧
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "采集失败", "无法获取摄像头画面")
            return
        
        # 检测棋盘格是否存在
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, _ = cv2.findChessboardCorners(gray, (self.XX, self.YY), None)
        if not ret_corners:
            QMessageBox.warning(self, "采集失败", "未检测到棋盘格，请调整后重试")
            return
        
        # 保存图像（文件名从0开始递增）
        image_path = os.path.join(self.save_dir, f"{self.capture_count}.jpg")
        cv2.imwrite(image_path, frame)
        
        # 获取机械臂位姿输入
        pose_str, ok = QInputDialog.getText(
            self, "输入机械臂位姿", 
            f"请输入第 {self.capture_count + 1} 组位姿 (x y z rx ry rz，单位：mm 和 deg):"
        )
        
        if ok and pose_str:
            try:
                # 解析输入
                pose = list(map(float, pose_str.split()))
                if len(pose) != 6:
                    raise ValueError("需要6个参数")
                
                # 转换单位：mm -> m，deg -> rad
                x, y, z, rx, ry, rz = pose
                x_m = x / 1000.0
                y_m = y / 1000.0
                z_m = z / 1000.0
                
                # 角度转弧度
                rx_rad = np.radians(rx)
                ry_rad = np.radians(ry)
                rz_rad = np.radians(rz)
                
                # 保存到位姿文件
                with open(self.poses_file, 'a') as f:
                    f.write(f"{x_m:.6f},{y_m:.6f},{z_m:.6f},{rx_rad:.6f},{ry_rad:.6f},{rz_rad:.6f}\n")
                
                self.capture_count += 1
                self.status_label.setText(f"已采集 {self.capture_count} 组数据 - 保存至: {os.path.basename(self.save_dir)} - 按 's' 继续采集")
                
            except ValueError as e:
                QMessageBox.warning(self, "输入错误", f"位姿格式错误: {str(e)}\n请使用空格分隔6个数值")
                # 删除已保存的图像
                if os.path.exists(image_path):
                    os.remove(image_path)
        else:
            # 用户取消输入，删除已保存的图像
            if os.path.exists(image_path):
                os.remove(image_path)
    
    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        if hasattr(self, 'cap'):
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = HandEyeCalibrationUI()
    window.show()
    sys.exit(app.exec_())
