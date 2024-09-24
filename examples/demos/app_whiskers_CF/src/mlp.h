#ifndef SRC_MLP_H_
#define SRC_MLP_H_
#include <stdint.h>
#include <stdbool.h>

// 假设这些是预定义的常量
#define HIDDEN_SIZE_1 32 // 第一层神经元数量
#define HIDDEN_SIZE_2 32 // 第二层神经元数量
#define HIDDEN_SIZE_3 32 // 第三层神经元数量


// dis_net 函数声明
void dis_net(StateWhisker *statewhisker);
void init_mlp_params(MLPParams* params_1, MLPParams* params_2);
void free_mlp_params(MLPParams* params_1, MLPParams* params_2);

typedef struct {
    float mean[INPUT_SIZE];
    float std[INPUT_SIZE];
    float W1[HIDDEN_SIZE_1][INPUT_SIZE];
    float b1[HIDDEN_SIZE_1];
    float W2[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
    float b2[HIDDEN_SIZE_2];
    float W3[HIDDEN_SIZE_3][HIDDEN_SIZE_2];
    float b3[HIDDEN_SIZE_3];
    float W4[HIDDEN_SIZE_3];
    float b4;
} MLPParams;


#endif // YOUR_FILE_NAME_H
