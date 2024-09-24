#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CF_whiskers_onboard.h"
#include "mlp.h"


void init_mlp_params(MLPParams* params_1, MLPParams* params_2) 
{
    // 为 MLPParams 的每个参数初始化赋值
    if (params_1 == NULL || params_2 == NULL) 
    {
        DEBUG_PRINT("MLP params are NULL!\n");
        return;
    }

    // 初始化 params_1
    for (int i = 0; i < INPUT_SIZE; i++) 
    {
        params_1->mean[i] = 0; // 设置初始值
        params_1->std[i] = 0;
    }
    for (int i = 0; i < HIDDEN_SIZE_1; i++) 
    {
        for (int j = 0; j < INPUT_SIZE; j++) {
            params_1->W1[i][j] = 0;
        }
        params_1->b1[i] = 0;
    }
    // 初始化 params_2，类似 params_1
    // ...
    
    DEBUG_PRINT("MLP parameters initialized.\n");
}

void free_mlp_params(MLPParams* params_1, MLPParams* params_2) 
{
    // 如果使用动态内存分配某些参数，可以在这里释放
    free(params_1);
    free(params_2);

    DEBUG_PRINT("MLP parameters freed.\n");
}


void normalization(const float* data, const float* mean, const float* std, float* normalized_data) 
{
    for (int i = 0; i < INPUT_SIZE; i++) 
    {
        normalized_data[i] = (data[i] - mean[i]) / std[i];
    }
}

void relu(const float* x, float* output, int size) 
{
    for (int i = 0; i < size; i++) {
        output[i] = fmax(0, x[i]);
    }
}

void mlp_inference(const float* input_data, 
                   const float W1[HIDDEN_SIZE_1][INPUT_SIZE], const float b1[HIDDEN_SIZE_1],
                   const float W2[HIDDEN_SIZE_2][HIDDEN_SIZE_1], const float b2[HIDDEN_SIZE_2],
                   const float W3[HIDDEN_SIZE_2][HIDDEN_SIZE_2], const float b3[HIDDEN_SIZE_2],
                   const float W4[HIDDEN_SIZE_2], const float b4,
                   float* output) 
{
    float z1[HIDDEN_SIZE_1];
    float a1[HIDDEN_SIZE_1];
    
    // 第一层前向传播
    for (int i = 0; i < HIDDEN_SIZE_1; i++) 
    {
        z1[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            z1[i] += input_data[j] * W1[i][j];
        }
        z1[i] += b1[i];
    }
    relu(z1, a1, HIDDEN_SIZE_1);
    
    float z2[HIDDEN_SIZE_2];
    float a2[HIDDEN_SIZE_2];
    
    // 第二层前向传播
    for (int i = 0; i < HIDDEN_SIZE_2; i++) 
    {
        z2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE_1; j++) 
        {
            z2[i] += a1[j] * W2[i][j];
        }
        z2[i] += b2[i];
    }
    relu(z2, a2, HIDDEN_SIZE_2);
    
    float z3[HIDDEN_SIZE_3];
    float a3[HIDDEN_SIZE_3];
    
    // 第三层前向传播
    for (int i = 0; i < HIDDEN_SIZE_3; i++) 
    {
        z3[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE_2; j++) 
        {
            z3[i] += a2[j] * W3[i][j];
        }
        z3[i] += b3[i];
    }
    relu(z3, a3, HIDDEN_SIZE_3);
    // 输出层前向传播
    *output = 0;
    for (int j = 0; j < HIDDEN_SIZE_3; j++) 
    {
        *output += a3[j] * W4[j];
    }
    *output += b4;
}

void process_whisker(const float* input_data, const float* mean, const float* std,
                     const float W1[HIDDEN_SIZE_1][INPUT_SIZE], const float b1[HIDDEN_SIZE_1],
                     const float W2[HIDDEN_SIZE_2][HIDDEN_SIZE_1], const float b2[HIDDEN_SIZE_2],
                     const float W3[HIDDEN_SIZE_2][HIDDEN_SIZE_2], const float b3[HIDDEN_SIZE_2],
                     const float W4[HIDDEN_SIZE_2], const float b4,
                     float* output) 
{
    float normalized_data[INPUT_SIZE];
    normalization(input_data, mean, std, normalized_data);
    mlp_inference(normalized_data, W1, b1, W2, b2, W3, b3, W4, b4, output);
}

void dis_net(StateWhisker *statewhisker, MLPParams* params_1, MLPParams* params_2) 
{
    float input_data_1[INPUT_SIZE] = {statewhisker->whisker1_1, statewhisker->whisker1_2, statewhisker->whisker1_3};
    float input_data_2[INPUT_SIZE] = {statewhisker->whisker2_1, statewhisker->whisker2_2, statewhisker->whisker2_3};

    process_whisker(input_data_1, params_1->mean, params_1->std, params_1->W1, params_1->b1, 
                    params_1->W2, params_1->b2, params_1->W3, params_1->b3, params_1->W4, params_1->b4, &statewhisker->mlpoutput_1);
    
    process_whisker(input_data_2, params_2->mean, params_2->std, params_2->W1, params_2->b1, 
                    params_2->W2, params_2->b2, params_2->W3, params_2->b3, params_2->W4, params_2->b4, &statewhisker->mlpoutput_2);
}