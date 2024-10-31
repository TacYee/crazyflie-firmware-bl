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
    gp->train_size = train_size;
    gp->kernel = inverse_multiquadric_kernel;
    gp->alpha = 1e-2; // 噪声

    // 动态分配内存
    gp->X_train = (float *)malloc(train_size * 2 * sizeof(float));
    gp->y_train = (float *)malloc(train_size * sizeof(float));
    gp->K_inv = (float *)malloc(train_size * train_size * sizeof(float)); // 修改为一维数组
    float *K_matrix_data = (float *)malloc(train_size * train_size * sizeof(float)); // 临时存储核矩阵 K

    for (int i = 0; i < train_size; ++i) 
    {
        gp->X_train[i * 2] = X_train[i * 2];
        gp->X_train[i * 2 + 1] = X_train[i * 2 + 1];
        gp->y_train[i] = y_train[i];
    }

    // 计算核矩阵 K(X_train, X_train) + αI
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
        // 错误处理，如果矩阵不可逆
        DEBUG_PRINT("Matrix is singular and cannot be inverted.\n");
    }

    free(K_matrix_data); // 释放 K_inv 分配的内存
}

// 预测新的输入
void gp_predict(const GaussianProcess *gp, const float *X_test, int test_size, float *y_pred, float *y_std) 
{
    // 定义矩阵结构
    arm_matrix_instance_f32 K_trans, K_test,K_inv_instance, y_train, y_mean;
    
    // 初始化 K(X_test, X_train) 矩阵
    float *K_trans_data = (float *)malloc(test_size * gp->train_size * sizeof(float));
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
    arm_mat_init_f32(&K_inv_instance, gp->train_size, gp->train_size, gp->K_inv); // 使用一维数组
    // 初始化 y_train 矩阵
    arm_mat_init_f32(&y_train, gp->train_size, 1, gp->y_train);


    arm_mat_init_f32(&y_mean, test_size, 1, y_pred); // 直接使用 y_pred 作为 y_mean 的数据存储

    // 计算预测均值 y_mean = K_trans * K_inv * y_train
    arm_matrix_instance_f32 K_trans_K_inv;
    float *K_trans_K_inv_data = (float *)malloc(test_size * gp->train_size * sizeof(float));
    arm_mat_init_f32(&K_trans_K_inv, test_size, gp->train_size, K_trans_K_inv_data);

    arm_mat_mult_f32(&K_trans, &K_inv_instance, &K_trans_K_inv);  // K_trans * K_inv
    arm_mat_mult_f32(&K_trans_K_inv, &y_train, &y_mean);  // K_trans_K_inv * y_train

    /*************************************************************************************************************/

    // 初始化 K(X_test, X_test) 矩阵
    float *K_test_data = (float *)malloc(test_size * test_size * sizeof(float));
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
    float *K_trans_T_data = (float *)malloc(gp->train_size * test_size * sizeof(float)); // K_trans_T 的数据存储
    arm_mat_init_f32(&K_trans_T, gp->train_size, test_size, K_trans_T_data);
    arm_mat_trans_f32(&K_trans, &K_trans_T); // 计算 K_trans 的转置
    // 计算方差 y_var = K_test - K_trans * K_inv * K_trans.T
    arm_matrix_instance_f32 K_trans_K_inv_K_transT;
    float *K_trans_K_inv_K_transT_data = (float *)malloc(test_size * test_size * sizeof(float));
    arm_mat_init_f32(&K_trans_K_inv_K_transT, test_size, test_size, K_trans_K_inv_K_transT_data);
    

    arm_mat_mult_f32(&K_trans_K_inv, &K_trans_T, &K_trans_K_inv_K_transT); // K_trans_K_inv * K_trans

    // 计算 y_var = K_test - K_trans_K_inv_K_transT
    arm_matrix_instance_f32 y_var;
    float *y_var_data = (float *)malloc(test_size * test_size * sizeof(float));
    arm_mat_init_f32(&y_var, test_size, test_size, y_var_data);
    
    arm_mat_sub_f32(&K_test, &K_trans_K_inv_K_transT, &y_var); // K_test - K_trans_K_inv_K_transT

    // 计算标准差
    for (int i = 0; i < test_size; ++i) {
        y_std[i] = sqrtf(y_var.pData[i * test_size + i]); // 取对角线元素计算标准差
    }

    // 释放动态分配的内存
    free(K_trans_data);
    free(K_test_data);
    free(K_trans_K_inv_data);
    free(K_trans_T_data); 
    free(K_trans_K_inv_K_transT_data);
    free(y_var_data);
}

// 清理高斯过程
void gp_free(GaussianProcess *gp) 
{
    free(gp->X_train); // 释放训练输入内存
    free(gp->y_train); // 释放训练输出内存
    free(gp->K_inv); // 释放指向行的指针
}
