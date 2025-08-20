
#眼在手外
import os  
import sys ,json 
import math  
currentdir = os.path.dirname(os.path.realpath(__file__))  
rootdir = os.path.dirname(os.path.dirname(currentdir))  
sys.path.insert(0, rootdir)  

import cv2  ,time
import numpy as np  
import trimesh  
import logging  
from scipy.spatial.transform import Rotation
# from arm import Arm  # 请确保更新为你自己的机械臂实现  
# from camera import Camera  # 请确保更新为你自己的相机实现  

# 设置日志记录器  
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
logger = logging.getLogger("CalibrationLogger")  

class Calibrator():  
    def __init__(  
            self,  
            calib_L: int,  
            calib_W: int,  
            calib_GRID: float,  
            # arm: Arm,  # 使用自定义机械臂类型  
            # camera: Camera  # 使用自定义相机类型   
    ):  
        """  
        Base calibrator class.  
        """  
        self.L: int = calib_L  
        self.W: int = calib_W  
        self.GRID: float = calib_GRID  
        # self.arm: Arm = arm  
        # self.camera: Camera = camera   
        # self.save_dir = os.path.join(os.getcwd(), "calibration_images")  # 保存图像的目录  
        
        # if not os.path.exists(self.save_dir):  
        #     os.makedirs(self.save_dir)  # 创建目录以保存图像  

    def calib(self, verbose: bool = True, save: bool = False, e2h: bool = True, cam_name:str="head") -> np.ndarray:  
        """  
        Hand Eye calibration. Currently support Eye-to-Hand calibration.  

        Args:  
            verbose: print calibration info verbosely.  
            save: save calibrated camera to base matrix as `calib.npy` in project rootdir.  
            e2h: calibrate with eye-to-hand or eye-in-hand.  

        Returns:  
            T: calibrated base to camera matrix, [4, 4].  
        """  
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  
        objp = np.zeros((self.W * self.L, 3), np.float32)  
        objp[:, :2] = np.mgrid[0: self.L, 0: self.W].T.reshape(-1, 2) * self.GRID  
        rvecs = []  
        tvecs = []  
        arm_transes = []  
        arm_rotmat_invs = []  
        i = 1   
        
        while i<25:  
            # rgb = self.camera.get_image()[0]  # 假设get_image返回RGB图像  
            # if rgb is None:  
            #     logger.warning("无法获取图像，继续下一轮...")   
            #     continue  # 无法获取图像，继续下一轮  
            # # 检查图像形状，如果是 RGBA（4 通道），则转换为 RGB（3 通道）  
            # if rgb.shape[2] == 4:  
            #     rgb = rgb[:, :, :3]  # 仅保留 RGB 通道  
            #     logger.info("RGB图像已从 RGBA 转换为 RGB。")  

            # fname=f'/home/taizun/WorkSpace/calibration/calbrate0424_1/{i}/{cam_name}.jpg'#{cam_name}
            fname=f'/home/taizun/WorkSpace/calibration/calibrate_data/20250521_head/{i}/{cam_name}.jpg'
            bgr = cv2.imread(fname) 
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  
            ret, corners = cv2.findChessboardCorners(gray, (self.L, self.W), None)  
            print(ret)
            if ret:  
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  
                cv2.drawChessboardCorners(bgr, (self.L, self.W), corners2, ret)  
                cv2.imshow('img', bgr)  
                key = cv2.waitKey(1)  
                
                # # 保存RGB图像  
                # rgb_filename = os.path.join(self.save_dir, f"rgb_image_{i}.png")  
                # cv2.imwrite(rgb_filename, rgb)  # 注意，这里保存的是原始RGB图像  
                # logger.info(f"保存RGB图像到: {rgb_filename}")  

                # 保存检测到的角点图像  
                # corners_filename = os.path.join(self.save_dir, f"/home/taizun/WorkSpace/calibration/cal_saved/corners_image_{i}.png")  
                corners_filename=f"/home/taizun/WorkSpace/calibration/calibrate_data/20250521_head_chessboard/corners_image_{i}_{cam_name}.png"
                cv2.imwrite(corners_filename, bgr)  # 保存包含角点的图像  
                logger.info(f"保存角点图像到: {corners_filename}")   
                #realsense l515
                k1=0.175372
                k2=-0.530182
                p1=-0.00120014
                p2=-0.00010845
                k3=0.488038
                distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)  
                color_intrinsics_mat=np.array([
                    [900.576, 0, 653.242],
                    [0,900.861, 370.284],
                    [0,0,1]])
                ret, rvec, tvec = cv2.solvePnP(objp, corners2, color_intrinsics_mat, distCoeffs)  # jibian
                if ret:  
                    i += 1  
                    rvecs.append(rvec)  
                    tvecs.append(tvec)  
                    if verbose:  
                        logger.info(f"测量 #{i}")  
                        logger.info(f"rvec: {rvec.flatten().tolist()}")  
                        logger.info(f"tvec: {tvec.flatten().tolist()}")  
                else: 
                    i=i+1 
                    if verbose:  
                        logger.warning("无法求解PnP，跳过当前帧。")   
                    continue  

                # # 获取机械臂当前位姿  
                # arm_pose = self.arm.get_pose()[1]  # Meter, radians, sxyz required 
                # arm_pose_list = arm_pose.split(',')    # 根据逗号分割  
                # arm_pose = list(map(float, arm_pose_list))  # 转换为浮点数列表  
                # # 将前3个值转换为米  
                # x_m = arm_pose[0]/ 1000.0  # 将 x 从 mm 转换为 m  
                # y_m = arm_pose[1]/ 1000.0  # 将 y 从 mm 转换为 m  
                # z_m = arm_pose[2] / 1000.0  # 将 z 从 mm 转换为 m  
                           
                # # # 获取旋转的角度（保持不变）  
                # roll = arm_pose[3]* (math.pi / 180) # 确保为浮点数  
                # pitch = arm_pose[4]* (math.pi / 180) # 确保为浮点数  
                # yaw = arm_pose[5] * (math.pi / 180)# 确保为浮点数  
                # roll = arm_pose[3]# 确保为浮点数  
                # pitch = arm_pose[4] # 确保为浮点数  
                # yaw = arm_pose[5] # 确保为浮点数 

                
                

                # # # 创建新的姿势列表，所有元素都是浮点数  
                # # arm_pose = [x_m, y_m, z_m, roll, pitch, yaw]  
                # # print(arm_pose)
                
                # if verbose:  
                #     logger.info(f"机械臂位姿: {arm_pose}")  
                # arm_rotmat_inv = trimesh.transformations.euler_matrix(  
                #     arm_pose[3], arm_pose[4], arm_pose[5]
                # )[:3, :3].T  
                # # print("345-----",float(arm_pose[3]), float(arm_pose[4]), float(arm_pose[5]) )
                # arm_transes.append(-arm_rotmat_inv @ arm_pose[:3])  
                # arm_rotmat_invs.append(arm_rotmat_inv)  

                # # 获取存储的机械臂位姿
                tcpfname=f'/home/taizun/WorkSpace/calibration/calibrate_data/20250521_head/{i-1}/arm_info.json'
                with open(tcpfname, 'r') as file:
                    data = json.load(file)

                RT=data["right_arm_info"]["RT"]
                RT_array=np.array(RT)
                arm_transe=RT_array[:3,3]

                arm_rotmat_inv=RT_array[:3,:3].T
                arm_transes.append(-arm_rotmat_inv @ arm_transe)
                arm_rotmat_invs.append(arm_rotmat_inv)

            else:
                i=i+1
            # time.sleep(0.05)

        if len(rvecs) < 3:  # 至少需要3个测量  
            logger.error("需要至少3个测量值来完成标定！")  
            return None  

        R, t = cv2.calibrateHandEye(arm_rotmat_invs, arm_transes, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_PARK)  
        T = np.eye(4)  
        T[:3, :3] = R  
        T[:3, 3] = t[:, 0]  

        if verbose:  
            logger.info(f"校正变换矩阵:\n{T}")  
        if save:  
            save_dir = os.path.join(os.getcwd(), "misc")  
            if not os.path.exists(save_dir):  
                os.makedirs(save_dir)  
            np.save(os.path.join(save_dir, f"calib_{cam_name}_1.npy"), T)    
        return T  


if __name__ == "__main__":  
    # # 模拟相机和机械臂的初始化  
    # camera = Camera()  # 请确保已有相机类  
    # camera.start()  # 启动相机  

    # arm = Arm(ip="192.168.201.1", port=29999)  # 请修改为你的机械臂IP  
    # try:  
    #     arm.connect()  # 确保连接到机械臂   
    #     logger.info("已成功连接到机械臂。")  
    # except Exception as e:  
    #     logger.error(f"机械臂连接失败: {str(e)}")  
    #     exit(1)  # 如果连接失败，则退出程序  
    calibrator = Calibrator(  
        calib_L=8,  
        calib_W=11,  
        calib_GRID=15/ 1000,  
        # arm=arm,  
        # camera=camera 
       
    )  
    T=calibrator.calib(save=False, cam_name="head")
    print(T)
    print(np.linalg.inv(T))