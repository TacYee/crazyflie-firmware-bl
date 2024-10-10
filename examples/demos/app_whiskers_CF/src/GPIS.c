#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "arm_math.h"  // 引入ARM Math库

#define MAX_TRAIN_SIZE 100  // 最大训练样本数量
#define MAX_TEST_SIZE 100    // 最大测试样本数量

typedef struct {
    float *X_train; // 动态分配
    float *y_train; // 动态分配
    int train_size; // 训练样本数量
    float **K_inv; // 动态分配的核矩阵的逆
    float (*kernel)(const float *, const float *, int, float); // 核函数
    float alpha;    // 噪声
} GaussianProcess;

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
    gp->alpha = 1e-10; // 噪声

    // 动态分配内存
    gp->X_train = (float *)malloc(train_size * 2 * sizeof(float));
    gp->y_train = (float *)malloc(train_size * sizeof(float));
    gp->K_inv = (float **)malloc(train_size * sizeof(float *));
    for (int i = 0; i < train_size; ++i) 
    {
        gp->K_inv[i] = (float *)malloc(train_size * sizeof(float));
    }

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
            gp->K_inv[i][j] = gp->kernel(&gp->X_train[i * 2], &gp->X_train[j * 2], 2, 2.0f);
        }
        gp->K_inv[i][i] += gp->alpha; // 加上噪声项
    }

    // 计算 K 的逆
    arm_matrix_instance_f32 K_matrix;
    arm_matrix_instance_f32 K_inv;

    // 初始化 K_matrix
    K_matrix.numRows = train_size;
    K_matrix.numCols = train_size;
    K_matrix.pData = (float*)gp->K_inv[0]; // 指向第一行

    // 为 K_inv 分配内存
    K_inv.numRows = train_size;
    K_inv.numCols = train_size;
    K_inv.pData = (float*)malloc(sizeof(float) * train_size * train_size); // 直接在这里分配内存

    // 计算 K 的逆
    arm_mat_inverse_f32(&K_matrix, &K_inv); // 计算 K 的逆

    // 将逆矩阵存入 gp->K_inv
    for (int i = 0; i < train_size; ++i) 
    {
        for (int j = 0; j < train_size; ++j) 
        {
            gp->K_inv[i][j] = K_inv.pData[i * train_size + j];
        }
    }

    free(K_inv.pData); // 释放 K_inv 分配的内存
}

// 预测新的输入
void gp_predict(const GaussianProcess *gp, const float *X_test, int test_size, float *y_pred, float *y_std) {
    float *K_trans = (float *)malloc(test_size * gp->train_size * sizeof(float)); // K(X_test, X_train)
    float *K_test = (float *)malloc(test_size * test_size * sizeof(float)); // K(X_test, X_test)

    for (int i = 0; i < test_size; ++i) {
        for (int j = 0; j < gp->train_size; ++j) {
            K_trans[i * gp->train_size + j] = gp->kernel(&X_test[i * 2], &gp->X_train[j * 2], 2, 2.0f);
        }
    }

    for (int i = 0; i < test_size; ++i) {
        for (int j = 0; j < test_size; ++j) {
            K_test[i * test_size + j] = gp->kernel(&X_test[i * 2], &X_test[j * 2], 2, 2.0f);
        }
    }

    // 计算预测均值
    for (int i = 0; i < test_size; ++i) 
    {
        y_pred[i] = 0.0f;
        for (int j = 0; j < gp->train_size; ++j) {
            y_pred[i] += K_trans[i * gp->train_size + j] * gp->K_inv[j][j]; // 使用逆矩阵的行 This is wrong, should input y_train
        }
    }

    // 计算方差
    for (int i = 0; i < test_size; ++i) {
        float y_var = K_test[i * test_size + i]; // K(X_test, X_test)
        for (int j = 0; j < gp->train_size; ++j) {
            y_var -= K_trans[i * gp->train_size + j] * gp->K_inv[j][j] * K_trans[i * gp->train_size + j];
        }
        y_std[i] = sqrtf(y_var);
    }

    // 释放动态分配的内存
    free(K_trans);
    free(K_test);
}

// 清理高斯过程
void gp_free(GaussianProcess *gp) {
    free(gp->X_train); // 释放训练输入内存
    free(gp->y_train); // 释放训练输出内存
    for (int i = 0; i < gp->train_size; ++i) {
        free(gp->K_inv[i]); // 释放每行的内存
    }
    free(gp->K_inv); // 释放指向行的指针
}

int main() {
    // 模拟一些训练数据
    float X_train[] = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f}; // 训练输入
    float y_train[] = {1.0f, 2.0f, 3.0f}; // 训练输出
    int train_size = 3; // 训练样本数量

    GaussianProcess gp;
    gp_fit(&gp, X_train, y_train, train_size);

    // 准备测试数据
    float X_test[] = {0.5f, 0.5f, 1.5f, 1.5f}; // 测试输入
    int test_size = 2;
    float y_pred[MAX_TEST_SIZE];
    float y_std[MAX_TEST_SIZE];

    // 进行预测
    gp_predict(&gp, X_test, test_size, y_pred, y_std);

    // 打印预测结果
    for (int i = 0; i < test_size; ++i) {
        printf("Prediction: %f, Std Dev: %f\n", y_pred[i], y_std[i]);
    }

    // 释放资源
    gp_free(&gp);

    return 0;
}