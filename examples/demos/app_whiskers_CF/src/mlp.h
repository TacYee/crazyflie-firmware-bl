#ifndef SRC_MLP_H_
#define SRC_MLP_H_
#include <stdint.h>
#include <stdbool.h>

// 假设这些是预定义的常量
#define INPUT_SIZE 3
#define HIDDEN_SIZE_1 32 // 第一层神经元数量
#define HIDDEN_SIZE_2 32 // 第二层神经元数量
#define HIDDEN_SIZE_3 32 // 第三层神经元数量

#include <arm_math.h> // 确保包含 CMSIS-DSP 头文件

typedef struct {
    float mean[INPUT_SIZE];                      // 均值
    float std[INPUT_SIZE];                       // 标准差
    arm_matrix_instance_f32 W1;                  // 第一层权重矩阵
    float b1[HIDDEN_SIZE_1];                     // 第一层偏置
    arm_matrix_instance_f32 W2;                  // 第二层权重矩阵
    float b2[HIDDEN_SIZE_2];                     // 第二层偏置
    arm_matrix_instance_f32 W3;                  // 第三层权重矩阵
    float b3[HIDDEN_SIZE_3];                     // 第三层偏置
    float W4[HIDDEN_SIZE_3];                     // 第四层权重，使用一维数组
    float b4;                                    // 第四层偏置
} MLPParams;


// dis_net 函数声明
void dis_net(StateWhisker *statewhisker, MLPParams* params_1, MLPParams* params_2);
void init_mlp_params(MLPParams* params_1, MLPParams* params_2);
void free_mlp_params(MLPParams* params_1, MLPParams* params_2);

#endif /* SRC_MLP_H_ */
