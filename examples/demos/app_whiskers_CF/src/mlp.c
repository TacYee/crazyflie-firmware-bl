#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CF_whiskers_onboard.h>
#include <mlp.h>


// 假设你有全局或外部定义的这些参数
extern float mean_1[INPUT_SIZE];
extern float std_1[INPUT_SIZE];
extern float mean_2[INPUT_SIZE];
extern float std_2[INPUT_SIZE];
extern float W1_1[HIDDEN_SIZE_1][INPUT_SIZE]; // 第一层权重
extern float b1_1[HIDDEN_SIZE_1]; // 第一层偏置
extern float W2_1[HIDDEN_SIZE_2][HIDDEN_SIZE_1]; // 第二层权重
extern float b2_1[HIDDEN_SIZE_2]; // 第二层偏置
extern float W3_1[HIDDEN_SIZE_3][HIDDEN_SIZE_3]; // 第三层权重
extern float b3_1[HIDDEN_SIZE_3]; // 第三层偏置
extern float W4_1[OUTPUT_SIZE][HIDDEN_SIZE_3]; // 输出层权重
extern float b4_1[OUTPUT_SIZE]; // 输出层偏置
extern float W1_2[HIDDEN_SIZE_1][INPUT_SIZE]; // 第一层权重
extern float b1_2[HIDDEN_SIZE_1]; // 第一层偏置
extern float W2_2[HIDDEN_SIZE_2][HIDDEN_SIZE_1]; // 第二层权重
extern float b2_2[HIDDEN_SIZE_2]; // 第二层偏置
extern float W3_2[HIDDEN_SIZE_3][HIDDEN_SIZE_3]; // 第三层权重
extern float b3_2[HIDDEN_SIZE_3]; // 第三层偏置
extern float W4_2[OUTPUT_SIZE][HIDDEN_SIZE_3]; // 输出层权重
extern float b4_2[OUTPUT_SIZE]; // 输出层偏置

void normalization(const float* data, const float* mean, const float* std, float* normalized_data) 
{
    for (int i = 0; i < INPUT_SIZE; i++) {
        normalized_data[i] = (data[i] - mean[i]) / std[i];
    }
}

void relu(const float* x, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = fmax(0, x[i]);
    }
}

void mlp_inference(const float* input_data, 
                   const float W1[HIDDEN_SIZE_1][INPUT_SIZE], const float b1[HIDDEN_SIZE_1],
                   const float W2[HIDDEN_SIZE_2][HIDDEN_SIZE_1], const float b2[HIDDEN_SIZE_2],
                   const float W3[HIDDEN_SIZE_2][HIDDEN_SIZE_2], const float b3[HIDDEN_SIZE_2],
                   const float W4[OUTPUT_SIZE][HIDDEN_SIZE_2], const float b4[OUTPUT_SIZE],
                   float* output) 
{
    float z1[HIDDEN_SIZE_1];
    float a1[HIDDEN_SIZE_1];
    
    // 第一层前向传播
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
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
    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
        z2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
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
        for (int j = 0; j < HIDDEN_SIZE_2; j++) {
            z3[i] += a2[j] * W3[i][j];
        }
        z3[i] += b3[i];
    }
    relu(z3, a3, HIDDEN_SIZE_3);
    // 输出层前向传播
    for (int i = 0; i < OUTPUT_SIZE; i++) 
    {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE_3; j++) {
            output[i] += a3[j] * W4[i][j];
        }
        output[i] += b4[i];
    }
}

void process_whisker(const float* input_data, const float* mean, const float* std,
                     const float W1[HIDDEN_SIZE_1][INPUT_SIZE], const float b1[HIDDEN_SIZE_1],
                     const float W2[HIDDEN_SIZE_2][HIDDEN_SIZE_1], const float b2[HIDDEN_SIZE_2],
                     const float W3[HIDDEN_SIZE_2][HIDDEN_SIZE_2], const float b3[HIDDEN_SIZE_2],
                     const float W4[OUTPUT_SIZE][HIDDEN_SIZE_2], const float b4[OUTPUT_SIZE],
                     float* output) {
    float normalized_data[INPUT_SIZE];
    normalization(input_data, mean, std, normalized_data);
    mlp_inference(normalized_data, W1, b1, W2, b2, W3, b3, W4, b4, output);
}

void dis_net(StateWhisker *statewhisker) {
    float input_data_1[INPUT_SIZE] = {statewhisker->whisker1_1, statewhisker->whisker1_2, statewhisker->whisker1_3};
    float input_data_2[INPUT_SIZE] = {statewhisker->whisker2_1, statewhisker->whisker2_2, statewhisker->whisker2_3};

    process_whisker(input_data_1, mean_1, std_1, W1_1, b1_1, W2_1, b2_1, W3_1, b3_1, W4_1, b4_1, statewhisker->mlpoutput_1);
    process_whisker(input_data_2, mean_2, std_2, W1_2, b1_2, W2_2, b2_2, W3_2, b3_2, W4_2, b4_2, statewhisker->mlpoutput_2);
}