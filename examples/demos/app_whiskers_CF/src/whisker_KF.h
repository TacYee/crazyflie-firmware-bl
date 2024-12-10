#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

typedef struct {
    float x_pre1;             // 状态估计
    float x_pre2; 
    float p_x_last;         // 上一位置 x
    float p_y_last;         // 上一位置 y
    float theta_last;           // 上一航向
    float P1;       // 协方差矩阵
    float P2; 
    float Q1;       // 过程噪声协方差矩阵
    float Q2; 
    float R;       // 测量噪声协方差矩阵
} KalmanFilterWhisker;
// 函数声明
void KF_init(KalmanFilterWhisker* kf, float initial_state1, float initial_state2, float initial_position[], 
             float initial_yaw, float initial_covariance, 
             float process_noise, float measurement_noise) ;
void KF_predict(KalmanFilterWhisker* kf, float position[], float yaw);
void KF_update(KalmanFilterWhisker* kf,  float z1, float z2) ;
void KF_get_estimate(KalmanFilterWhisker* kf, float* estimate1, float* estimate2);
void KF_data_receive(StateWhisker *statewhisker, KalmanFilterWhisker *kf);

#endif // KALMAN_FILTER_H
