#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 初始化卡尔曼滤波器
void KF_init(KalmanFilterWhisker* kf, float initial_state, float initial_position[], 
             float initial_yaw, float initial_covariance, 
             float process_noise, float measurement_noise) 
{
    kf->x_pre = 234 - initial_state;
    kf->p_x_last = -(initial_position[1] * 1000);
    kf->p_y_last = initial_position[0] * 1000;
    kf->Θ_last = initial_yaw;

    kf->P = initial_covariance;
    kf->Q = process_noise;
    kf->R = measurement_noise;
}


// 预测步骤
void KF_predict(KalmanFilterWhisker* kf, float position[], float yaw) 
{
    float p_x = -(position[1] * 1000);
    float p_y = position[0] * 1000;
    float relative_distance = sqrtf(powf(p_x - kf->p_x_last, 2) + powf(p_y - kf->p_y_last, 2));
    float new_relative_rad = (yaw - kf->Θ_last) * (M_PI / 180.0); // 转换为弧度

    // 预测状态
    kf->x_pre = kf->x_pre + relative_distance * sinf(kf->Θ_last * (M_PI / 180.0) - atan2f((p_y - kf->p_y_last), (p_x - kf->p_x_last)));
    kf->x_pre /= cosf(new_relative_rad);
    
    // 预测协方差矩阵
    kf->P += kf->Q;
    
    // 更新上一状态
    kf->Θ_last = yaw;
    kf->p_x_last = p_x;
    kf->p_y_last = p_y;
}

void KF_update(KalmanFilterWhisker* kf, float z) {
    // 计算卡尔曼增益 K
    float P_plus_R = kf->P + kf->R; // 预测协方差加测量噪声
    float K = kf->P / P_plus_R; // 卡尔曼增益

    // 更新状态估计
    kf->x_pre += K * (z - kf->x_pre); // 更新状态估计

    // 更新协方差矩阵
    kf->P *= (1 - K); // 更新协方差
}


// 获取当前状态估计
float KF_get_estimate(KalmanFilterWhisker* kf) 
{
    return 234 - kf->x_pre;
}

void KF_data_receive(StateWhisker *statewhisker, KalmanFilterWhisker *kf1, KalmanFilterWhisker *kf2) {
    // 进行预测
    KF_predict(kf1, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw);
    KF_predict(kf2, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw);
    
    // 更新滤波器
    KF_update(kf1, statewhisker->mlpoutput_1);
    KF_update(kf2, statewhisker->mlpoutput_2);
    
    // 将 KF 的估计值保存到 statewhisker
    statewhisker->KFoutput_1 = KF_get_estimate(kf1);
    statewhisker->KFoutput_2 = KF_get_estimate(kf2);
}