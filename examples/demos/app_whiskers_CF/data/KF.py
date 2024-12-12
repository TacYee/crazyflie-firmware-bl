import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = initial_state  # 状态估计
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵

    def predict(self):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre
        
        # 预测协方差矩阵：P_k^- = P_k-1 + Q
        self.P_minus = self.P + self.Q

    def update(self, z):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return self.x_pre
    

class KalmanFilterUA:
    def __init__(self, initial_state, initial_covariance, process_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = initial_state  # 状态估计
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = None  # 测量噪声协方差矩阵

    def predict(self):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre
        
        # 预测协方差矩阵：P_k^- = P_k-1 + Q
        self.P_minus = self.P + self.Q

    def update(self, z, measurement_noise):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        self.R  =measurement_noise
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return self.x_pre
    
class KalmanFilterFLAT:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 234 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre + (p_x - self.p_x_last) * np.sin(np.deg2rad(self.Θ_last)) + (self.p_y_last - p_y) * np.cos(np.deg2rad(self.Θ_last))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus = self.P + self.Q
        self.Θ_minus = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre
        
class KalmanFilterUAFLAT:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 234 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = None  # 测量噪声协方差矩阵

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre + (p_x - self.p_x_last) * np.sin(np.deg2rad(self.Θ_last)) + (self.p_y_last - p_y) * np.cos(np.deg2rad(self.Θ_last))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus = self.P + self.Q
        self.Θ_minus = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z, measurement_noise):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        self.R  = measurement_noise
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre
    
class KalmanFilterFLAT2:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 240 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        self.relative_distance = np.sqrt((p_x-self.p_x_last)**2+(p_y-self.p_y_last)**2)
        self.new_relative_rad = np.deg2rad(yaw - self.Θ_last)
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre + self.relative_distance * np.sin(np.deg2rad(self.Θ_last) - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last)))
        self.x_pre_minus = (self.x_pre_minus) / np.cos(self.new_relative_rad)
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus = self.P + self.Q
        self.Θ_last = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 240 - self.x_pre

class KalmanFilterFLAT2_1:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 234 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        self.relative_distance = np.sqrt((p_x-self.p_x_last)**2+(p_y-self.p_y_last)**2)
        self.new_relative_rad = np.deg2rad(yaw - self.Θ_last)
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre + self.relative_distance * np.sin(np.deg2rad(self.Θ_last) - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last)))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus = self.P + self.Q
        self.Θ_last = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre
    
class KalmanFilterFLAT2_guido:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 234 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵
        
        self.z_last = None  # 初始化上一个测量值为 None

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        self.relative_distance = np.sqrt((p_x - self.p_x_last)**2 + (p_y - self.p_y_last)**2)
        self.new_relative_rad = np.deg2rad(yaw - self.Θ_last)

        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre + self.relative_distance * np.sin(np.deg2rad(self.Θ_last) - np.arctan2(p_y - self.p_y_last, p_x - self.p_x_last))
        self.x_pre_minus /= np.cos(self.new_relative_rad)

        # 预测协方差矩阵：P_k^- = P_k-1 + Q
        self.P_minus = self.P + self.Q
        self.Θ_last = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 检查当前测量值是否小于上一个测量值
        if self.z_last is not None and z < self.z_last:
            self.Q = 0.05  # 将过程噪声设为 0
        
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

        # 更新上一个测量值
        self.z_last = z

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre
    
class KalmanFilterUAFLAT2:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 234 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = None  # 测量噪声协方差矩阵

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.relative_distance = np.sqrt((p_x-self.p_x_last)**2+(p_y-self.p_y_last)**2)
        self.new_relative_rad = np.deg2rad(yaw - self.Θ_last)
        self.x_pre_minus = self.x_pre + self.relative_distance * np.sin(np.deg2rad(self.Θ_last) - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last)))
        self.x_pre_minus = (self.x_pre_minus) / np.cos(self.new_relative_rad)
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus = self.P + self.Q
        self.Θ_last = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z, measurement_noise):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        self.R  = measurement_noise
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre
    
class KalmanFilterFLAT3:
    def __init__(self, initial_state1, initial_state2, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise, fading_factor):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre1 = 234 - initial_state1  # 状态估计
        self.x_pre2 = 234 - initial_state2  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P1 = initial_covariance  # 协方差矩阵
        self.P2 = initial_covariance  # 协方差矩阵
        self.Q1 = process_noise  # 过程噪声协方差矩阵
        self.Q2 = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵
        self.lambda_fading = fading_factor  # 衰减因子
    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        #无人机与墙壁法线夹角的预测值弧度
        self.relative_rad = np.arctan((self.x_pre1 - self.x_pre2) / 50)

        self.x_pre = (self.x_pre1 - 25 * np.tan(self.relative_rad) + self.x_pre2 + 25 * np.tan(self.relative_rad))/2
        #墙壁在world frame下的角度预测值弧度
        self.wall_rad = np.deg2rad(self.Θ_last) - self.relative_rad 
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        #无人机移动的position的相对距离
        self.relative_distance = np.sqrt((p_x-self.p_x_last)**2+(p_y-self.p_y_last)**2)
        #新的position和yaw进入后的new relative rad
        self.new_relative_rad = np.deg2rad(yaw) - self.wall_rad
        #在assumption下无人机与估计墙壁的距离
        self.distance_towall = (self.relative_distance * np.sin(self.wall_rad - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last))))/np.cos(self.relative_rad) + self.x_pre
        # reorentation后的预测状态：
        self.x_pre_minus = (self.distance_towall * np.cos(self.relative_rad)) / np.cos(self.new_relative_rad)
        self.x_pre_minus1 = (self.x_pre_minus + 25 * np.tan(self.new_relative_rad))
        self.x_pre_minus2 = (self.x_pre_minus - 25 * np.tan(self.new_relative_rad))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus1 = self.lambda_fading * self.P1 + self.Q1
        self.P_minus2 = self.lambda_fading * self.P2 + self.Q2
        self.p_x_last = p_x
        self.p_y_last = p_y
        self.Θ_last = yaw
        # print(self.relative_rad)
        # print(f"relative_rad = {self.relative_rad}")
        # print(self.distance_towall)
        print(f"wall_rad = {self.wall_rad}")
        print(self.x_pre_minus)

    def update(self, z1, z2):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K1 = self.P_minus1 @ np.linalg.inv(self.P_minus1 + self.R)
        self.K2 = self.P_minus2 @ np.linalg.inv(self.P_minus2 + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre1 = self.x_pre_minus1 + self.K1 @ (z1 - self.x_pre_minus1)
        self.x_pre2 = self.x_pre_minus2 + self.K2 @ (z2 - self.x_pre_minus2)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P1.shape[0])  # 单位矩阵
        self.P1 = (I - self.K1) @ self.P_minus1
        self.P2 = (I - self.K2) @ self.P_minus2

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre1, 234 - self.x_pre2, 234 - self.x_pre
    
class KalmanFilterFLAT3_seperate:
    def __init__(self, initial_state1, initial_state2, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise, fading_factor):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre1 = 240 - initial_state1  # 状态估计
        self.x_pre2 = 240 - initial_state2  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P1 = initial_covariance  # 协方差矩阵
        self.P2 = initial_covariance  # 协方差矩阵
        self.Q1 = process_noise  # 过程噪声协方差矩阵
        self.Q2 = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵
        self.lambda_fading = fading_factor  # 衰减因子
    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        #无人机与墙壁法线夹角的预测值弧度
        self.relative_rad = 0

        #墙壁在world frame下的角度预测值弧度
        self.wall_rad = np.deg2rad(self.Θ_last) - self.relative_rad 
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        #无人机移动的position的相对距离
        self.relative_distance = np.sqrt((p_x-self.p_x_last)**2+(p_y-self.p_y_last)**2)
        #新的position和yaw进入后的new relative rad
        self.new_relative_rad = np.deg2rad(yaw) - self.wall_rad
        #在assumption下无人机与估计墙壁的距离
        self.distance_towall1 = (self.relative_distance * np.sin(self.wall_rad - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last))))/np.cos(self.relative_rad) + self.x_pre1
        self.distance_towall2 = (self.relative_distance * np.sin(self.wall_rad - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last))))/np.cos(self.relative_rad) + self.x_pre2
        # reorentation后的预测状态：
        self.x_pre_minus1 = (self.distance_towall1 * np.cos(self.relative_rad)) 
        self.x_pre_minus2 = (self.distance_towall2 * np.cos(self.relative_rad))
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus1 = self.lambda_fading * self.P1 + self.Q1
        self.P_minus2 = self.lambda_fading * self.P2 + self.Q2
        self.p_x_last = p_x
        self.p_y_last = p_y
        self.Θ_last = yaw
        # print(self.relative_rad)
        # print(f"relative_rad = {self.relative_rad}")
        # print(self.distance_towall)
        print(f"wall_rad = {self.wall_rad}")

    def update(self, z1, z2):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K1 = self.P_minus1 @ np.linalg.inv(self.P_minus1 + self.R)
        self.K2 = self.P_minus2 @ np.linalg.inv(self.P_minus2 + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre1 = self.x_pre_minus1 + self.K1 @ (z1 - self.x_pre_minus1)
        self.x_pre2 = self.x_pre_minus2 + self.K2 @ (z2 - self.x_pre_minus2)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P1.shape[0])  # 单位矩阵
        self.P1 = (I - self.K1) @ self.P_minus1
        self.P2 = (I - self.K2) @ self.P_minus2

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 240 - self.x_pre1, 240 - self.x_pre2
    
class KalmanFilterUAFLAT3:
    def __init__(self, initial_state1, initial_state2, initial_position, initial_yaw, initial_covariance, process_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre1 = 234 - initial_state1  # 状态估计
        self.x_pre2 = 234 - initial_state2  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P1 = initial_covariance  # 协方差矩阵
        self.P2 = initial_covariance  # 协方差矩阵
        self.Q1 = process_noise  # 过程噪声协方差矩阵
        self.Q2 = process_noise  # 过程噪声协方差矩阵
    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        #无人机与墙壁法线夹角的预测值弧度
        self.relative_rad = np.arctan((self.x_pre1 - self.x_pre2) / 50)

        self.x_pre = self.x_pre1 - 25 * np.tan(self.relative_rad) 
        #墙壁在world frame下的角度预测值弧度
        self.wall_rad = np.deg2rad(self.Θ_last) - self.relative_rad 
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        #无人机移动的position的相对距离
        self.relative_distance = np.sqrt((p_x-self.p_x_last)**2+(p_y-self.p_y_last)**2)
        #新的position和yaw进入后的new relative rad
        self.new_relative_rad = np.deg2rad(yaw) - self.wall_rad
        #在assumption下无人机与估计墙壁的距离
        self.distance_towall = (self.relative_distance * np.sin(self.wall_rad - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last))))/np.cos(self.relative_rad) + self.x_pre
        # reorentation后的预测状态：
        self.x_pre_minus = (self.distance_towall * np.cos(self.relative_rad)) / np.cos(self.new_relative_rad)
        self.x_pre_minus1 = (self.x_pre_minus + 25 * np.tan(self.new_relative_rad))
        self.x_pre_minus2 = (self.x_pre_minus - 25 * np.tan(self.new_relative_rad))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus1 = self.P1 + self.Q1
        self.P_minus2 = self.P2 + self.Q2
        self.p_x_last = p_x
        self.p_y_last = p_y
        self.Θ_last = yaw
        # print(self.relative_rad)
        # print(f"relative_rad = {self.relative_rad}")
        # print(self.distance_towall)

    def update(self, z1, z2, measurement_noise1, measurement_noise2):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.R1  = measurement_noise1
        self.R2  = measurement_noise2
        self.K1 = self.P_minus1 @ np.linalg.inv(self.P_minus1 + self.R1)
        self.K2 = self.P_minus2 @ np.linalg.inv(self.P_minus2 + self.R2)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre1 = self.x_pre_minus1 + self.K1 @ (z1 - self.x_pre_minus1)
        self.x_pre2 = self.x_pre_minus2 + self.K2 @ (z2 - self.x_pre_minus2)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P1.shape[0])  # 单位矩阵
        self.P1 = (I - self.K1) @ self.P_minus1
        self.P2 = (I - self.K2) @ self.P_minus2

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre1, 234 - self.x_pre2, 234 - self.x_pre
    
class KalmanFilterUA2FLAT3:
    def __init__(self, initial_state1, initial_state2, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise, fading_factor):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre1 = 234 - initial_state1  # 状态估计
        self.x_pre2 = 234 - initial_state2  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P1 = initial_covariance  # 协方差矩阵
        self.P2 = initial_covariance  # 协方差矩阵
        self.Q1 = process_noise  # 过程噪声协方差矩阵
        self.Q2 = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵
        self.lambda_fading = fading_factor  # 衰减因子
    def predict(self, position, yaw, measurement_noise1, measurement_noise2):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        #无人机与墙壁法线夹角的预测值弧度
        factor = measurement_noise1 / (measurement_noise1 + measurement_noise2)
        self.relative_rad = np.arctan((self.x_pre1 - self.x_pre2) / 50)

        self.x_pre = (1 - factor) * (self.x_pre1 - 25 * np.tan(self.relative_rad)) + factor * (self.x_pre2 + 25 * np.tan(self.relative_rad))
        #墙壁在world frame下的角度预测值弧度
        self.wall_rad = np.deg2rad(self.Θ_last) - self.relative_rad 
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        #无人机移动的position的相对距离
        self.relative_distance = np.sqrt((p_x-self.p_x_last)**2+(p_y-self.p_y_last)**2)
        #新的position和yaw进入后的new relative rad
        self.new_relative_rad = np.deg2rad(yaw) - self.wall_rad
        #在assumption下无人机与估计墙壁的距离
        self.distance_towall = (self.relative_distance * np.sin(self.wall_rad - np.arctan((p_y-self.p_y_last)/(p_x-self.p_x_last))))/np.cos(self.relative_rad) + self.x_pre
        # reorentation后的预测状态：
        self.x_pre_minus = (self.distance_towall * np.cos(self.relative_rad)) / np.cos(self.new_relative_rad)
        self.x_pre_minus1 = (self.x_pre_minus + 25 * np.tan(self.new_relative_rad))
        self.x_pre_minus2 = (self.x_pre_minus - 25 * np.tan(self.new_relative_rad))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus1 = self.lambda_fading * self.P1 + self.Q1
        self.P_minus2 = self.lambda_fading * self.P2 + self.Q2
        self.p_x_last = p_x
        self.p_y_last = p_y
        self.Θ_last = yaw
        # print(self.relative_rad)
        # print(f"relative_rad = {self.relative_rad}")
        # print(self.distance_towall)
        print(f"wall_rad = {self.wall_rad}")
        print(self.x_pre_minus)

    def update(self, z1, z2):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K1 = self.P_minus1 @ np.linalg.inv(self.P_minus1 + self.R)
        self.K2 = self.P_minus2 @ np.linalg.inv(self.P_minus2 + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre1 = self.x_pre_minus1 + self.K1 @ (z1 - self.x_pre_minus1)
        self.x_pre2 = self.x_pre_minus2 + self.K2 @ (z2 - self.x_pre_minus2)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P1.shape[0])  # 单位矩阵
        self.P1 = (I - self.K1) @ self.P_minus1
        self.P2 = (I - self.K2) @ self.P_minus2

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre1, 234 - self.x_pre2, 234 - self.x_pre
    
class KalmanFilterFLAT3_guido:
    def __init__(self, initial_state1, initial_state2, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise, fading_factor):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre1 = 234 - initial_state1  # 状态估计
        self.x_pre2 = 234 - initial_state2  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P1 = initial_covariance  # 协方差矩阵
        self.P2 = initial_covariance  # 协方差矩阵
        self.Q1 = process_noise  # 过程噪声协方差矩阵
        self.Q2 = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵
        self.lambda_fading = fading_factor  # 衰减因子
        
        self.z_last1 = None  # 上一个测量值 z1
        self.z_last2 = None  # 上一个测量值 z2

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        # 无人机与墙壁法线夹角的预测值弧度
        self.relative_rad = np.arctan((self.x_pre1 - self.x_pre2) / 50)

        self.x_pre = (self.x_pre1 - 25 * np.tan(self.relative_rad) + self.x_pre2 + 25 * np.tan(self.relative_rad)) / 2
        # 墙壁在world frame下的角度预测值弧度
        self.wall_rad = np.deg2rad(self.Θ_last) - self.relative_rad 
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        # 无人机移动的position的相对距离
        self.relative_distance = np.sqrt((p_x - self.p_x_last) ** 2 + (p_y - self.p_y_last) ** 2)
        # 新的position和yaw进入后的new relative rad
        self.new_relative_rad = np.deg2rad(yaw) - self.wall_rad
        # 在assumption下无人机与估计墙壁的距离
        self.distance_towall = (self.relative_distance * np.sin(self.wall_rad - np.arctan((p_y - self.p_y_last) / (p_x - self.p_x_last)))) / np.cos(self.relative_rad) + self.x_pre
        # reorientation后的预测状态：
        self.x_pre_minus = (self.distance_towall * np.cos(self.relative_rad)) / np.cos(self.new_relative_rad)
        self.x_pre_minus1 = (self.x_pre_minus + 25 * np.tan(self.new_relative_rad))
        self.x_pre_minus2 = (self.x_pre_minus - 25 * np.tan(self.new_relative_rad))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus1 = self.lambda_fading * self.P1 + self.Q1
        self.P_minus2 = self.lambda_fading * self.P2 + self.Q2
        self.p_x_last = p_x
        self.p_y_last = p_y
        self.Θ_last = yaw
        
        print(f"wall_rad = {self.wall_rad}")
        print(self.x_pre_minus)

    def update(self, z1, z2):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 检查当前测量值是否小于上一个测量值
        if self.z_last1 is not None and z1 < self.z_last1:
            self.Q1 = 0.05  # 将过程噪声设为 0
        if self.z_last2 is not None and z2 < self.z_last2:
            self.Q2 = 0.05  # 将过程噪声设为 0
        
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K1 = self.P_minus1 @ np.linalg.inv(self.P_minus1 + self.R)
        self.K2 = self.P_minus2 @ np.linalg.inv(self.P_minus2 + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre1 = self.x_pre_minus1 + self.K1 @ (z1 - self.x_pre_minus1)
        self.x_pre2 = self.x_pre_minus2 + self.K2 @ (z2 - self.x_pre_minus2)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P1.shape[0])  # 单位矩阵
        self.P1 = (I - self.K1) @ self.P_minus1
        self.P2 = (I - self.K2) @ self.P_minus2
        
        # 更新上一个测量值
        self.z_last1 = z1
        self.z_last2 = z2

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre1, 234 - self.x_pre2, 234 - self.x_pre