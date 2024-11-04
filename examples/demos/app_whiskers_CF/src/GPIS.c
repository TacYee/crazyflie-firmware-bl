#include "GPIS.h"
#include <stdio.h>
#include <math.h>
#include "arm_math.h"
#include "debug.h"



// 高斯过程中的逆多二次核函数
float inverse_multiquadric_kernel(const float *X, const float *Y, int dim, float c) 
{
    float dist_sq = 0.0f;
    for (int i = 0; i < dim; ++i) 
    {
        dist_sq += (X[i] - Y[i]) * (X[i] - Y[i]);
    }
    return 1.0f / sqrtf(dist_sq + c * c);
}

// 拟合训练数据
void gp_fit(GaussianProcess *gp, const float *X_train, const float *y_train, int train_size) 
{
    // 检查训练样本数量是否超过最大值
    if (train_size > MAX_TRAIN_SIZE) {
        DEBUG_PRINT("Training size exceeds maximum limit.\n");
        return;
    }

    gp->train_size = train_size;
    gp->kernel = inverse_multiquadric_kernel;
    gp->alpha = 1e-2; // 噪声

    for (int i = 0; i < train_size; ++i) 
    {
        gp->X_train[i * 2] = X_train[i * 2];
        gp->X_train[i * 2 + 1] = X_train[i * 2 + 1];
        gp->y_train[i] = y_train[i];
    }

    // 计算核矩阵 K(X_train, X_train) + αI
    float K_matrix_data[MAX_TRAIN_SIZE * MAX_TRAIN_SIZE]; // 临时存储核矩阵 K
    for (int i = 0; i < train_size; ++i) 
    {
        for (int j = 0; j < train_size; ++j) 
        {
            K_matrix_data[i * train_size + j] = gp->kernel(&gp->X_train[i * 2], &gp->X_train[j * 2], 2, 2.0f);
        }
        K_matrix_data[i * train_size + i] += gp->alpha; // 对角线上加上噪声项 α
    }

    // 计算 K 的逆
    arm_matrix_instance_f32 K_matrix;
    arm_matrix_instance_f32 K_inv;
    arm_mat_init_f32(&K_matrix, train_size, train_size, K_matrix_data);
    arm_mat_init_f32(&K_inv, train_size, train_size, gp->K_inv);
    
    // 计算 K_matrix 的逆，并存储到 gp->K_inv
    if (arm_mat_inverse_f32(&K_matrix, &K_inv) != ARM_MATH_SUCCESS) 
    {
        DEBUG_PRINT("Matrix is singular and cannot be inverted.\n");
    }
}

// 预测新的输入
void gp_predict(const GaussianProcess *gp, const float *X_test, int test_size, float *y_pred, float *y_std) 
{
    // 定义矩阵结构
    arm_matrix_instance_f32 K_trans, K_test, K_inv_instance, y_train, y_mean;

    // 初始化 K(X_test, X_train) 矩阵
    float K_trans_data[MAX_TRAIN_SIZE * 2]; // 使用静态数组
    arm_mat_init_f32(&K_trans, test_size, gp->train_size, K_trans_data);

    // 计算 K_trans 矩阵 K(X_test, X_train)
    for (int i = 0; i < test_size; ++i) 
    {
        for (int j = 0; j < gp->train_size; ++j) 
        {
            K_trans.pData[i * gp->train_size + j] = gp->kernel(&X_test[i * 2], &gp->X_train[j * 2], 2, 2.0f);
        }
    }

    // 初始化 K_inv 矩阵
    arm_mat_init_f32(&K_inv_instance, gp->train_size, gp->train_size, (float*)gp->K_inv);
    arm_mat_init_f32(&y_train, gp->train_size, 1, (float*)gp->y_train);

    arm_mat_init_f32(&y_mean, test_size, 1, y_pred); // 直接使用 y_pred 作为 y_mean 的数据存储

    // 计算预测均值 y_mean = K_trans * K_inv * y_train
    arm_matrix_instance_f32 K_trans_K_inv;
    float K_trans_K_inv_data[MAX_TRAIN_SIZE * 2]; // 使用静态数组
    arm_mat_init_f32(&K_trans_K_inv, test_size, gp->train_size, K_trans_K_inv_data);

    arm_mat_mult_f32(&K_trans, &K_inv_instance, &K_trans_K_inv);  // K_trans * K_inv
    arm_mat_mult_f32(&K_trans_K_inv, &y_train, &y_mean);  // K_trans_K_inv * y_train

    // 初始化 K(X_test, X_test) 矩阵
    float K_test_data[MAX_TRAIN_SIZE * MAX_TRAIN_SIZE]; // 使用静态数组
    arm_mat_init_f32(&K_test, test_size, test_size, K_test_data);
    
    // 计算 K_test 矩阵 K(X_test, X_test)
    for (int i = 0; i < test_size; ++i) 
    {
        for (int j = 0; j < test_size; ++j) 
        {
            K_test.pData[i * test_size + j] = gp->kernel(&X_test[i * 2], &X_test[j * 2], 2, 2.0f);
        }
    }

    // 计算 K_trans 的转置
    arm_matrix_instance_f32 K_trans_T;
    float K_trans_T_data[MAX_TRAIN_SIZE * 2]; // 使用静态数组
    arm_mat_init_f32(&K_trans_T, gp->train_size, test_size, K_trans_T_data);
    arm_mat_trans_f32(&K_trans, &K_trans_T); // 计算 K_trans 的转置

    // 计算方差 y_var = K_test - K_trans * K_inv * K_trans.T
    arm_matrix_instance_f32 K_trans_K_inv_K_transT;
    float K_trans_K_inv_K_transT_data[MAX_TRAIN_SIZE * MAX_TRAIN_SIZE]; // 使用静态数组
    arm_mat_init_f32(&K_trans_K_inv_K_transT, test_size, test_size, K_trans_K_inv_K_transT_data);
    
    arm_mat_mult_f32(&K_trans_K_inv, &K_trans_T, &K_trans_K_inv_K_transT); // K_trans_K_inv * K_trans

    // 计算 y_var = K_test - K_trans_K_inv_K_transT
    arm_matrix_instance_f32 y_var;
    float y_var_data[MAX_TRAIN_SIZE * MAX_TRAIN_SIZE]; // 使用静态数组
    arm_mat_init_f32(&y_var, test_size, test_size, y_var_data);
    
    arm_mat_sub_f32(&K_test, &K_trans_K_inv_K_transT, &y_var); // K_test - K_trans_K_inv_K_transT

    // 计算标准差
    for (int i = 0; i < test_size; ++i) {
        y_std[i] = sqrtf(y_var.pData[i * test_size + i]); // 取对角线元素计算标准差
    }

    // 不再需要释放内存
}
