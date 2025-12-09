import os
import cv2
import numpy as np
import yaml
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import sys
import threading
import subprocess
import shutil

class HandEyeCalibrationCLI:
    def __init__(self):
        self.load_config()
        self.init_camera()
        self.init_data_storage()  # 自动生成保存目录
        self.capture_count = 0
        self.frame = None  # 存储最新帧
        self.detected_chessboard = False  # 棋盘格检测状态
        self.running = True  # 程序运行标志
    
    def load_config(self):
        """加载标定板配置参数"""
        try:
            with open("config.yaml", 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            self.XX = data.get("checkerboard_args").get("XX")  # 标定板长度方向角点个数
            self.YY = data.get("checkerboard_args").get("YY")  # 标定板宽度方向角点个数
            self.L = data.get("checkerboard_args").get("L")    # 标定板格子长度(米)
            
            self.W = data.get("W", 1280)  # 默认1280
            self.H = data.get("H", 720)   # 默认720
            # 设置亚像素角点检测参数
            self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
            
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            print("使用默认配置参数")
            self.XX, self.YY, self.L = 9, 6, 0.02  # 默认值
            self.W, self.H = 1280, 720
    
    def init_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(0)  # 默认摄像头
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        
        if not self.cap.isOpened():
            print("无法打开摄像头，请检查设备连接")
            sys.exit(1)
        
        # 验证实际分辨率
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头实际分辨率: {actual_width}x{actual_height}")
    
    def get_next_save_dir(self):
        """自动生成下一个保存目录（格式：dataYYYYMMDDXX）"""
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_hand_data")
        today = datetime.now().strftime("%Y%m%d")
        base_name = f"data{today}"
        
        # 确保基础目录存在
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # 查找当前日期下已存在的组号
        existing_dirs = []
        for dir_name in os.listdir(self.base_dir):
            if dir_name.startswith(base_name) and len(dir_name) == len(base_name) + 2:
                suffix = dir_name[-2:]
                if suffix.isdigit():
                    existing_dirs.append(int(suffix))
        
        # 确定下一个组号
        next_num = max(existing_dirs) + 1 if existing_dirs else 1
        next_suffix = f"{next_num:02d}"
        new_dir = os.path.join(self.base_dir, f"{base_name}{next_suffix}")
        
        return new_dir
    
    def init_data_storage(self):
        """初始化数据存储目录"""
        self.save_dir = self.get_next_save_dir()
        
        # 创建目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 初始化位姿文件路径
        self.poses_file = os.path.join(self.save_dir, "poses.txt")
        
        # 清空已有poses.txt文件
        with open(self.poses_file, 'w') as f:
            pass
        
        print(f"\n保存目录已创建: {self.save_dir}")
        print("=" * 60)
    
    def camera_display_thread(self):
        """摄像头画面显示线程（独立于命令行输入）"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame.copy()
                
                # 检测棋盘格
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_corners, corners = cv2.findChessboardCorners(gray, (self.XX, self.YY), None)
                self.detected_chessboard = ret_corners
                
                # 绘制角点和状态文字
                display_frame = frame.copy()
                status_text = f"已采集: {self.capture_count} 组 | 保存目录: {os.path.basename(self.save_dir)}"
                
                if ret_corners:
                    # 亚像素优化并绘制角点（绿色）
                    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(display_frame, (self.XX, self.YY), corners2, ret_corners)
                    cv2.putText(display_frame, "✅ Find Chessboard", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 绿色
                else:
                    # 未检测到棋盘格（红色）
                    cv2.putText(display_frame, "❌ Not Find Chessboard", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 红色
                
                # 显示状态信息
                cv2.putText(display_frame, status_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "命令行: s=采集 | i=眼在手上 | o=眼在手外 | q=退出", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示画面
                cv2.imshow("手眼标定数据采集", display_frame)
            
            # 保持窗口响应（仅用于关闭窗口）
            if cv2.waitKey(1) & 0xFF == ord('x'):  # 隐藏的退出快捷键，用于紧急关闭
                self.running = False
                break
        
        # 释放摄像头和窗口
        self.cap.release()
        cv2.destroyAllWindows()
    
    def start_capture(self):
        """启动程序主逻辑"""
        print("手眼标定数据采集程序")
        print("=" * 60)
        print("操作说明:")
        print("  在命令行中输入以下指令并回车：")
        print("  - 's' 或 'S' : 采集当前帧和机械臂位姿（需先检测到棋盘格）")
        print("  - 'i' 或 'I' : 眼在手上标定计算（相机相对于机械臂末端）")
        print("  - 'o' 或 'O' : 眼在手外标定计算（相机相对于机械臂基座）")
        print("  - 'q' 或 'Q' : 退出程序")
        print("  - 直接回车 : 刷新检测状态，不执行操作")
        print("=" * 60)
        print("提示: 请关注cv2显示窗口中的棋盘格检测状态")
        print("=" * 60)
        
        # 启动摄像头显示线程
        display_thread = threading.Thread(target=self.camera_display_thread, daemon=True)
        display_thread.start()
        
        # 命令行输入循环
        while self.running:
            try:
                # 命令行输入操作指令
                user_input = input("\n请输入操作指令 (s=采集, i=眼在手上计算, o=眼在手外计算, q=退出): ").strip()
                
                if user_input.lower() == 'q':
                    # 退出程序
                    print("\n正在退出程序...")
                    self.running = False
                    break
                
                elif user_input.lower() == 's':
                    # 采集数据
                    if self.frame is None:
                        print("❌ 错误: 未获取到摄像头画面，请稍后再试")
                        continue
                    
                    if self.detected_chessboard:
                        self.capture_data()
                    else:
                        print("❌ 错误: 未检测到棋盘格，无法采集数据")
                        print("  请调整摄像头位置或标定板角度后重试")
                
                elif user_input.lower() == 'i':
                    # 眼在手上计算
                    self.compute_in_hand()
                
                elif user_input.lower() == 'o':
                    # 眼在手外计算
                    self.compute_to_hand()
                
                elif user_input == '':
                    # 直接回车，刷新检测状态
                    status = "✅ 已检测到" if self.detected_chessboard else "❌ 未检测到"
                    print(f"当前状态: {status} 棋盘格 | 已采集: {self.capture_count} 组")
                
                else:
                    # 无效输入
                    print(f"❌ 无效指令: '{user_input}'")
                    print("  请输入 's' 采集, 'i' 眼在手上计算, 'o' 眼在手外计算, 或 'q' 退出")
            
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ 操作出错: {str(e)}")
                continue
        
        # 等待显示线程结束
        display_thread.join(timeout=2.0)
        print("\n资源已释放，程序正常退出")
    
    def capture_data(self):
        """采集数据（保存图像和位姿）"""
        # 保存图像（使用最新帧）
        image_path = os.path.join(self.save_dir, f"{self.capture_count}.jpg")
        cv2.imwrite(image_path, self.frame)
        print(f"\n📷 已保存图像: {os.path.basename(image_path)}")
        
        # 命令行输入机械臂位姿
        print(f"\n📝 请输入第 {self.capture_count + 1} 组机械臂位姿")
        print("格式说明: x y z rx ry rz （单位：mm 和 deg，用空格分隔）")
        print("示例: 100.5 200.3 300.0 10.2 20.5 30.1")
        print("输入 'cancel' 可取消本次采集")
        
        while True:
            pose_input = input("请输入位姿: ").strip()
            
            if pose_input.lower() == 'cancel':
                # 取消采集，删除已保存的图像
                if os.path.exists(image_path):
                    os.remove(image_path)
                print("❌ 本次采集已取消")
                return
            
            if not pose_input:
                print("⚠️  警告: 输入不能为空，请重新输入")
                continue
            
            try:
                # 解析输入
                pose = list(map(float, pose_input.split()))
                if len(pose) != 6:
                    raise ValueError(f"需要6个参数，实际输入了{len(pose)}个")
                
                x, y, z, rx, ry, rz = pose
                
                # 单位转换：mm -> m，deg -> rad
                x_m = x / 1000.0
                y_m = y / 1000.0
                z_m = z / 1000.0
                rx_rad = np.radians(rx)
                ry_rad = np.radians(ry)
                rz_rad = np.radians(rz)
                
                # 保存到位姿文件
                with open(self.poses_file, 'a') as f:
                    f.write(f"{x_m:.6f},{y_m:.6f},{z_m:.6f},{rx_rad:.6f},{ry_rad:.6f},{rz_rad:.6f}\n")
                
                self.capture_count += 1
                print(f"\n✅ 第 {self.capture_count} 组数据采集成功！")
                print(f"   原始位姿（mm, deg）: {x:.2f}, {y:.2f}, {z:.2f}, {rx:.2f}, {ry:.2f}, {rz:.2f}")
                print(f"   转换后（m, rad）: {x_m:.6f}, {y_m:.6f}, {z_m:.6f}, {rx_rad:.6f}, {ry_rad:.6f}, {rz_rad:.6f}")
                break
                
            except ValueError as e:
                print(f"❌ 输入错误: {str(e)}")
                print("请重新输入，或输入 'cancel' 取消")
    
    def compute_in_hand(self):
        """眼在手上标定计算"""
        try:
            print("\n🔧 开始眼在手上标定计算...")
            print("计算相机相对于机械臂末端的位姿")
            print("=" * 60)
            
            # 检查数据目录是否存在
            current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_hand_data")
            if not os.path.exists(current_path):
                print("❌ 错误: 未找到 eye_hand_data 目录")
                print("  请先采集数据再进行计算")
                return
            
            # 查找最新的数据文件夹
            from libs.auxiliary import find_latest_data_folder
            latest_folder = find_latest_data_folder(current_path)
            if not latest_folder:
                print("❌ 错误: 未找到有效的数据文件夹")
                print("  请先采集数据再进行计算")
                return
            
            data_path = os.path.join(current_path, latest_folder)
            images_path = data_path
            file_path = os.path.join(data_path, "poses.txt")
            
            # 检查必要文件
            if not os.path.exists(file_path):
                print(f"❌ 错误: 未找到位姿文件 {file_path}")
                return
            
            # 检查图片数量
            images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
            if len(images) == 0:
                print("❌ 错误: 未找到图片文件")
                return
            
            print(f"📁 使用数据目录: {latest_folder}")
            print(f"📸 找到 {len(images)} 张图片")
            print(f"📄 位姿文件: {os.path.basename(file_path)}")
            print("=" * 60)
            
            # 运行计算
            import compute_in_hand
            rotation_matrix, translation_vector = compute_in_hand.in_hand_calib(images_path, file_path)
            
            # 转换为四元数
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            x, y, z = translation_vector.flatten()
            
            print("=" * 60)
            print("✅ 眼在手上标定计算完成！")
            print("=" * 60)
            print(f"旋转矩阵:\n{rotation_matrix}")
            print(f"\n平移向量 (m): [{x:.6f}, {y:.6f}, {z:.6f}]")
            print(f"\n四元数 (x,y,z,w): [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
            
            # 保存结果
            result_file = os.path.join(data_path, "eye_in_hand_result.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write("眼在手上标定结果\n")
                f.write("=" * 40 + "\n")
                f.write(f"旋转矩阵:\n{rotation_matrix}\n\n")
                f.write(f"平移向量 (m): [{x:.6f}, {y:.6f}, {z:.6f}]\n\n")
                f.write(f"四元数 (x,y,z,w): [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]\n")
            
            print(f"\n💾 结果已保存到: {result_file}")
            
        except Exception as e:
            print(f"❌ 计算过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def compute_to_hand(self):
        """眼在手外标定计算"""
        try:
            print("\n🔧 开始眼在手外标定计算...")
            print("计算相机相对于机械臂基座的位姿")
            print("=" * 60)
            
            # 检查数据目录是否存在
            current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_hand_data")
            if not os.path.exists(current_path):
                print("❌ 错误: 未找到 eye_hand_data 目录")
                print("  请先采集数据再进行计算")
                return
            
            # 查找最新的数据文件夹
            from libs.auxiliary import find_latest_data_folder
            latest_folder = find_latest_data_folder(current_path)
            if not latest_folder:
                print("❌ 错误: 未找到有效的数据文件夹")
                print("  请先采集数据再进行计算")
                return
            
            data_path = os.path.join(current_path, latest_folder)
            images_path = data_path
            file_path = os.path.join(data_path, "poses.txt")
            
            # 检查必要文件
            if not os.path.exists(file_path):
                print(f"❌ 错误: 未找到位姿文件 {file_path}")
                return
            
            # 检查图片数量
            images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
            if len(images) == 0:
                print("❌ 错误: 未找到图片文件")
                return
            
            print(f"📁 使用数据目录: {latest_folder}")
            print(f"📸 找到 {len(images)} 张图片")
            print(f"📄 位姿文件: {os.path.basename(file_path)}")
            print("=" * 60)
            
            # 运行计算
            import compute_to_hand
            rotation_matrix, translation_vector = compute_to_hand.to_hand_calib(images_path, file_path)
            
            # 转换为四元数
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            x, y, z = translation_vector.flatten()
            
            print("=" * 60)
            print("✅ 眼在手外标定计算完成！")
            print("=" * 60)
            print(f"旋转矩阵:\n{rotation_matrix}")
            print(f"\n平移向量 (m): [{x:.6f}, {y:.6f}, {z:.6f}]")
            print(f"\n四元数 (x,y,z,w): [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
            
            # 保存结果
            result_file = os.path.join(data_path, "eye_to_hand_result.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write("眼在手外标定结果\n")
                f.write("=" * 40 + "\n")
                f.write(f"旋转矩阵:\n{rotation_matrix}\n\n")
                f.write(f"平移向量 (m): [{x:.6f}, {y:.6f}, {z:.6f}]\n\n")
                f.write(f"四元数 (x,y,z,w): [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]\n")
            
            print(f"\n💾 结果已保存到: {result_file}")
            
        except Exception as e:
            print(f"❌ 计算过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        calibration = HandEyeCalibrationCLI()
        calibration.start_capture()
    except Exception as e:
        print(f"\n❌ 程序异常: {str(e)}")
        # 确保资源释放
        if 'calibration' in locals():
            calibration.running = False
            if hasattr(calibration, 'cap'):
                calibration.cap.release()
        cv2.destroyAllWindows()
        print("资源已释放，程序退出")
