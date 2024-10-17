#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "arm_math.h"  // 引入ARM Math库

// 定义 float16 类型
typedef uint16_t float16;

#define MAX_TRAIN_SIZE 100  // 最大训练样本数量
#define MAX_TEST_SIZE 100    // 最大测试样本数量

typedef struct {
    float16 *X_train; // 动态分配
    float16 *y_train; // 动态分配
    int train_size; // 训练样本数量
    float16 **K_inv; // 动态分配的核矩阵的逆
    float (*kernel)(const float16 *, const float16 *, int, float); // 核函数
    float alpha;    // 噪声
} GaussianProcess;

// 转换 float16 到 float
float float16_to_float(float16 h) {
    // Implement float16 to float conversion
    // This is a simplified version. Actual conversion will depend on the specific float16 representation.
    return (float)(h >> 8) * (1 << (h & 0xFF));  // Example conversion
}

// 转换 float 到 float16
float16 float_to_float16(float f) {
    // Implement float to float16 conversion
    // This is a simplified version. Actual conversion will depend on the specific float16 representation.
    return (float16)(f * 256.0f);  // Example conversion
}

// 高斯过程中的逆多二次核函数
float inverse_multiquadric_kernel(const float16 *X, const float16 *Y, int dim, float c) 
{
    float dist_sq = 0.0f;
    for (int i = 0; i < dim; ++i) 
    {
        float x_i = float16_to_float(X[i]);
        float y_i = float16_to_float(Y[i]);
        dist_sq += (x_i - y_i) * (x_i - y_i);
    }
    return 1.0f / sqrtf(dist_sq + c * c);
}

// 拟合训练数据
void gp_fit(GaussianProcess *gp, const float16 *X_train, const float16 *y_train, int train_size) 
{
    gp->train_size = train_size;
    gp->kernel = inverse_multiquadric_kernel;
    gp->alpha = 1e-10; // 噪声

    // 动态分配内存
    gp->X_train = (float16 *)malloc(train_size * 2 * sizeof(float16));
    gp->y_train = (float16 *)malloc(train_size * sizeof(float16));
    gp->K_inv = (float16 **)malloc(train_size * sizeof(float16 *));
    for (int i = 0; i < train_size; ++i) 
    {
        gp->K_inv[i] = (float16 *)malloc(train_size * sizeof(float16));
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
            gp->K_inv[i][j] = float_to_float16(gp->kernel(&gp->X_train[i * 2], &gp->X_train[j * 2], 2, 2.0f));
        }
        gp->K_inv[i][i] += float_to_float16(gp->alpha); // 加上噪声项
    }

    // 计算 K 的逆
    arm_matrix_instance_f32 K_matrix;
    arm_matrix_instance_f32 K_inv;

    // 初始化 K_matrix
    K_matrix.numRows = train_size;
    K_matrix.numCols = train_size;
    K_matrix.pData = (float*)malloc(sizeof(float) * train_size * train_size); // 需要为 float 类型分配内存

    // 转换 K_inv 为 float 类型
    for (int i = 0; i < train_size; ++i) 
    {
        for (int j = 0; j < train_size; ++j) 
        {
            K_matrix.pData[i * train_size + j] = float16_to_float(gp->K_inv[i][j]);
        }
    }

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
            gp->K_inv[i][j] = float_to_float16(K_inv.pData[i * train_size + j]);
        }
    }   

    free(K_inv.pData); // 释放 K_inv 分配的内存
}

// 预测新的输入
void gp_predict(const GaussianProcess *gp, const float16 *X_test, int test_size, float16 *y_pred, float16 *y_std) {
    float16 *K_trans = (float16 *)malloc(test_size * gp->train_size * sizeof(float16)); // K(X_test, X_train)
    float16 *K_test = (float16 *)malloc(test_size * test_size * sizeof(float16)); // K(X_test, X_test)

    for (int i = 0; i < test_size; ++i) 
    {
        for (int j = 0; j < gp->train_size; ++j) {
            K_trans[i * gp->train_size + j] = float_to_float16(gp->kernel(&X_test[i * 2], &gp->X_train[j * 2], 2, 2.0f));
        }
    }

    for (int i = 0; i < test_size; ++i) 
    {
        for (int j = 0; j < test_size; ++j) {
            K_test[i * test_size + j] = float_to_float16(gp->kernel(&X_test[i * 2], &X_test[j * 2], 2, 2.0f));
        }
    }

    // 计算预测均值
    for (int i = 0; i < test_size; ++i) 
    {
        y_pred[i] = 0.0f;
        for (int j = 0; j < gp->train_size; ++j) 
        {
            float K_inv_yj = 0.0;
            for (int k = 0; k < gp->train_size; k++) 
            {
                K_inv_yj += float16_to_float(gp->K_inv[j][k]) * float16_to_float(gp->y_train[k]);
            }
            y_pred[i] += float_to_float16(K_trans[i * gp->train_size + j] * K_inv_yj); 
        }
    }

    // 计算方差
    for (int i = 0; i < test_size; ++i) {
        float y_var = float16_to_float(K_test[i * test_size + i]); // K(X_test, X_test)
        
        // Subtract the contribution of the training data
        for (int j = 0; j < gp->train_size; j++) {
            float K_trans_ij = float16_to_float(K_trans[i * gp->train_size + j]); // K_trans[i, j]
            float temp_sum = 0.0;

            // Calculate the contribution of K_inv * K_trans[j, :]
            for (int k = 0; k < gp->train_size; k++) {
                temp_sum += float16_to_float(gp->K_inv[j][k]) * float16_to_float(K_trans[k * test_size + i]); // K_trans[k, i]
            }

            y_var -= K_trans_ij * temp_sum; // Subtract the contribution
        }
        y_std[i] = float_to_float16(sqrt(y_var));
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