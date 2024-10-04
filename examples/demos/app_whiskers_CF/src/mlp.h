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

// dis_net 函数声明
void dis_net(StateWhisker *statewhisker);

#endif /* SRC_MLP_H_ */
