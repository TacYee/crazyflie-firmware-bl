#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

typedef struct {
    float x_pre;             // 状态估计
    float p_x_last;         // 上一位置 x
    float p_y_last;         // 上一位置 y
    float Θ_last;           // 上一航向
    float P;       // 协方差矩阵
    float Q;       // 过程噪声协方差矩阵
    float R;       // 测量噪声协方差矩阵
} KalmanFilterFLAT2;
// 函数声明
void KF_init(KalmanFilter *kf, float initial_state, float initial_covariance, float process_noise, float measurement_noise);
void KF_predict(KalmanFilter *kf);
void KF_update(KalmanFilter *kf, float z);
float KF_get_estimate(KalmanFilter *kf);
void KF_data_receive(KalmanFilter *kf1, KalmanFilter *kf2, float z1, float z2);

#endif // KALMAN_FILTER_H
