/*
 * wall_follower_multi_ranger_onboard.c
 *
 *  Created on: Sep 9, 2024
 *      Author: Chaoxiang Ye

 */

#include "CF_whiskers_onboard.h"
#include <math.h>
#include <stdlib.h>
#include "mlp.h"
#include "whisker_KF.h"
#include "debug.h"
#include "log.h"
#include "GPIS.h"

#define DATA_SIZE 100

static float firstRun = false;
static float firstRunPreprocess = false;
static float maxSpeed = 0.2f;
static float maxTurnRate = 25.0f;
static float CF_THRESHOLD1 = 20.0f;
static float CF_THRESHOLD2 = 20.0f;
static float MIN_THRESHOLD1 = 25.0f;
static float MAX_THRESHOLD1 = 100.0f;
static float MIN_THRESHOLD2 = 25.0f;
static float MAX_THRESHOLD2 = 100.0f;
static float MAX_FILTERTHRESHOLD1 = 10.0f;
static float MAX_FILTERTHRESHOLD2 = 10.0f;
static float StartTime;
static const float waitForStartSeconds = 10.0f;
static float stateStartTime;
static int is_KF_initialized = 0; // 0 表示未初始化, 1 表示已初始化
static int CF_count = 150;
static int backward_count = 150;
static int rotate_count = 180;
// 定义高斯过程模型和相关变量
GaussianProcess gp_model;  // 高斯过程模型实例
int current_train_index = 0;  // 当前已收集的训练数据索引
float X_train[MAX_TRAIN_SIZE * 2];  // 训练输入数据，假设每个样本有两个特征
float y_train[MAX_TRAIN_SIZE];      // 训练输出数据
int is_gp_trained = 0;              // 标记 GP 是否已训练
// static MLPParams params_1; // 第一个 MLP 参数
// static MLPParams params_2; // 第二个 MLP 参数
static KalmanFilterWhisker kf1;
static KalmanFilterWhisker kf2;
static float process_noise = 0.000001;
static float measurement_noise = 1;
static float initial_covariance = 0.1;

static StateCF stateCF = hover;
float timeNow = 0.0f;

void ProcessWhiskerInit(StateWhisker *statewhisker) 
{
    for (int i = 0; i < 6; i++) 
    {
        statewhisker->zi[i][0] = 0.0f;  // Initialize filter state
        statewhisker->zi[i][1] = 0.0f;
    }
    
    statewhisker->count = 0;
    statewhisker->preprocesscount = 0;
    
    // 1st order butterworth filter 0.05-1hz
    statewhisker->b[0] = 0.05639124;
    statewhisker->b[1] = 0.0f;
    statewhisker->b[2] = -0.05639124;
    statewhisker->a[0] = 1.0f;
    statewhisker->a[1] = -1.88647164;
    statewhisker->a[2] = 0.88721752;
    firstRunPreprocess = true;
    DEBUG_PRINT("Initialize preprocessing parameters.\n");
}

void update_statistics(StateWhisker *statewhisker, float data, int index) 
{
    float x = (float)statewhisker->count;
    statewhisker->sum_x[index] += x;
    statewhisker->sum_y[index] += data;
    statewhisker->sum_x_squared[index] += x * x;
    statewhisker->sum_xy[index] += x * data;
}


void calculate_parameters(StateWhisker *statewhisker) 
{
    int n = DATA_SIZE;
    for (int i = 0; i < 6; i++) 
    {
        statewhisker->slopes[i] = (n * statewhisker->sum_xy[i] - statewhisker->sum_x[i] * statewhisker->sum_y[i]) /
                             (n * statewhisker->sum_x_squared[i] - statewhisker->sum_x[i] * statewhisker->sum_x[i]);
        statewhisker->intercepts[i] = (statewhisker->sum_y[i] - statewhisker->slopes[i] * statewhisker->sum_x[i]) / n;
    }
}


void apply_bandpass_filter(float data, float *zi, float *filtered_data, float *b, float *a) 
{
    float input = data;
    float output = b[0] * input + zi[0];
    zi[0] = b[1] * input - a[1] * output + zi[1];
    zi[1] = b[2] * input - a[2] * output;
    *filtered_data = output;
}



void process_data(StateWhisker *statewhisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) 
{
    float residuals[6];
    residuals[0] = whisker1_1 - (statewhisker->slopes[0] * statewhisker->count + statewhisker->intercepts[0]);
    residuals[1] = whisker1_2 - (statewhisker->slopes[1] * statewhisker->count + statewhisker->intercepts[1]);
    residuals[2] = whisker1_3 - (statewhisker->slopes[2] * statewhisker->count + statewhisker->intercepts[2]);
    residuals[3] = whisker2_1 - (statewhisker->slopes[3] * statewhisker->count + statewhisker->intercepts[3]);
    residuals[4] = whisker2_2 - (statewhisker->slopes[4] * statewhisker->count + statewhisker->intercepts[4]);
    residuals[5] = whisker2_3 - (statewhisker->slopes[5] * statewhisker->count + statewhisker->intercepts[5]);

    apply_bandpass_filter(residuals[0], statewhisker->zi[0], &statewhisker->whisker1_1, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[1], statewhisker->zi[1], &statewhisker->whisker1_2, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[2], statewhisker->zi[2], &statewhisker->whisker1_3, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[3], statewhisker->zi[3], &statewhisker->whisker2_1, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[4], statewhisker->zi[4], &statewhisker->whisker2_2, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[5], statewhisker->zi[5], &statewhisker->whisker2_3, statewhisker->b, statewhisker->a);
}

void ProcessDataReceived(StateWhisker *statewhisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) 
{
    if (firstRunPreprocess)
    {
        update_statistics(statewhisker, whisker1_1, 0);
        update_statistics(statewhisker, whisker1_2, 1);
        update_statistics(statewhisker, whisker1_3, 2);
        update_statistics(statewhisker, whisker2_1, 3);
        update_statistics(statewhisker, whisker2_2, 4);
        update_statistics(statewhisker, whisker2_3, 5);
        statewhisker->count++;
        if (statewhisker->count == DATA_SIZE) 
        {
            calculate_parameters(statewhisker);
            for (int i = 0; i < 6; i++) 
            {
                statewhisker->sum_x[i] = 0.0f;
                statewhisker->sum_y[i] = 0.0f;
                statewhisker->sum_x_squared[i] = 0.0f;
                statewhisker->sum_xy[i] = 0.0f;
            }
            firstRunPreprocess = false;
            DEBUG_PRINT("First Apply linear fitting parameters.\n");
        }
    }
    else 
    {   
        update_statistics(statewhisker, whisker1_1, 0);
        update_statistics(statewhisker, whisker1_2, 1);
        update_statistics(statewhisker, whisker1_3, 2);
        update_statistics(statewhisker, whisker2_1, 3);
        update_statistics(statewhisker, whisker2_2, 4);
        update_statistics(statewhisker, whisker2_3, 5);
        statewhisker->count++;
        statewhisker->preprocesscount++;
        process_data(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > MAX_FILTERTHRESHOLD1 || statewhisker->whisker2_1 > MAX_FILTERTHRESHOLD2)
        {
            for (int i = 0; i < 6; i++) 
            {
                statewhisker->sum_x[i] = 0.0f;
                statewhisker->sum_y[i] = 0.0f;
                statewhisker->sum_x_squared[i] = 0.0f;
                statewhisker->sum_xy[i] = 0.0f;
            }
            statewhisker->preprocesscount = 0;
        }
        else if (statewhisker->preprocesscount == DATA_SIZE) 
        {
            calculate_parameters(statewhisker);
            for (int i = 0; i < 6; i++) 
            {
                statewhisker->sum_x[i] = 0.0f;
                statewhisker->sum_y[i] = 0.0f;
                statewhisker->sum_x_squared[i] = 0.0f;
                statewhisker->sum_xy[i] = 0.0f;
                statewhisker->zi[i][0] = 0.0f;  // Initialize filter state
                statewhisker->zi[i][1] = 0.0f;
            }
            statewhisker->preprocesscount = 0;
            DEBUG_PRINT("Apply linear fitting parameters.\n");
        }
    }
}

void FSMInit(float MIN_THRESHOLD1_input, float MAX_THRESHOLD1_input, 
             float MIN_THRESHOLD2_input, float MAX_THRESHOLD2_input, 
             float maxSpeed_input, float maxTurnRate_input, StateCF initState)
{
  MIN_THRESHOLD1 = MIN_THRESHOLD1_input;
  MAX_THRESHOLD1 = MAX_THRESHOLD1_input;
  MIN_THRESHOLD2 = MIN_THRESHOLD2_input;
  MAX_THRESHOLD2 = MAX_THRESHOLD2_input;
  maxSpeed = maxSpeed_input;
  maxTurnRate = maxTurnRate_input;
  firstRun = true;
  stateCF = initState;
  CF_count = 150;
  DEBUG_PRINT("Initialize FSM parameters.\n");
}

static StateCF transition(StateCF newState)
{
  stateStartTime = timeNow;
  return newState;
}

StateCF FSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
        }
        break;
    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->whisker1_1 < MIN_THRESHOLD1 && statewhisker->whisker2_1 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && statewhisker->whisker2_1 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->whisker1_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && statewhisker->whisker2_1 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->whisker1_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}


StateCF MLPFSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);

        if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {   
            dis_net(statewhisker);
        }else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            dis_net(statewhisker);
            statewhisker->mlpoutput_2 = 0.0f;
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            dis_net(statewhisker);
            statewhisker->mlpoutput_1 = 0.0f; 
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");

            statewhisker->mlpoutput_1 = 0.0f; 
            statewhisker->mlpoutput_2 = 0.0f;
            DEBUG_PRINT("Exiting CF state, resources released.\n");
        }
        break;
    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->mlpoutput_1 < MIN_THRESHOLD1 && statewhisker->mlpoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->mlpoutput_1 && statewhisker->mlpoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->mlpoutput_2 && statewhisker->mlpoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->mlpoutput_1 && statewhisker->mlpoutput_1 > MIN_THRESHOLD1 && statewhisker->mlpoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->mlpoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->mlpoutput_2 && statewhisker->mlpoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->mlpoutput_1 && statewhisker->mlpoutput_1 > MIN_THRESHOLD1 && statewhisker->mlpoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->mlpoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->mlpoutput_2 && statewhisker->mlpoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}

StateCF FSM_data(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        CF_count--;
        if (CF_count < 0)
        {
            stateCF = transition(backward);
            DEBUG_PRINT("CF finished. Flyingbackward.\n");
        }
        else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
        }
        break;

    case backward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        break;

    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->whisker1_1 < MIN_THRESHOLD1 && statewhisker->whisker2_1 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && statewhisker->whisker2_1 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->whisker1_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && statewhisker->whisker2_1 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->whisker1_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    case backward:
        cmdVelXTemp = -1.0f * maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}

StateCF KFMLPFSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);

        if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {        
            if (!is_KF_initialized)
            {
                dis_net(statewhisker);
                KF_init(&kf1, 30.0f, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw, initial_covariance, process_noise, measurement_noise);//here need a interface
                KF_init(&kf2, 30.0f, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw, initial_covariance, process_noise, measurement_noise);
                is_KF_initialized = 1; // 标记已初始化
                statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
                statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            }
            else
            {
                dis_net(statewhisker);
                KF_data_receive(statewhisker, &kf1, &kf2);
            }

        }else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            dis_net(statewhisker);
            statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_2 = 0.0f;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 2 loses contact. Reset KF.\n");
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            dis_net(statewhisker);
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 1 loses contact. Reset KF.\n");
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
            is_KF_initialized = 0;
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            DEBUG_PRINT("Exiting CF state, resources released.\n");
        }
        break;
    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}


StateCF KFMLPFSM_EXP(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        CF_count--;
        if (CF_count < 0)
        {
            stateCF = transition(backward);
            DEBUG_PRINT("CF finished. Flyingbackward.\n");
            backward_count = 150;
        }
        else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {        
            if (!is_KF_initialized)
            {
                dis_net(statewhisker);
                KF_init(&kf1, 30.0f, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw, initial_covariance, process_noise, measurement_noise);//here need a interface
                KF_init(&kf2, 30.0f, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw, initial_covariance, process_noise, measurement_noise);
                is_KF_initialized = 1; // 标记已初始化
                statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
                statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
                if (current_train_index < MAX_TRAIN_SIZE && CF_count%30 == 0) 
                {
                    // 保存状态输入特征 (statewhisker->KFoutput_1)
                    X_train[current_train_index * 2] = (234.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f) * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x * 1000;   // 第一个特征
                    X_train[current_train_index * 2 + 1] = - (statewhisker->p_y * 1000) - (234.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f) * sinf(statewhisker->yaw * (M_PI_F / 180.0f)); // 第二个特征

                    // 保存输出标签 y (设为 0)
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  // 更新训练数据索引
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
            }
            else
            {
                dis_net(statewhisker);
                KF_data_receive(statewhisker, &kf1, &kf2);
                if (current_train_index < MAX_TRAIN_SIZE && CF_count % 30 == 0) 
                {
                    // 保存状态输入特征 (statewhisker->KFoutput_1)
                    X_train[current_train_index * 2] = (234.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f) * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x * 1000;   // 第一个特征
                    X_train[current_train_index * 2 + 1] = - (statewhisker->p_y * 1000) - (234.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f) * sinf(statewhisker->yaw * (M_PI_F / 180.0f)); // 第二个特征

                    // 保存输出标签 y (设为 0)
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  // 更新训练数据索引
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
            }

        }else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            dis_net(statewhisker);
            statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_2 = 0.0f;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 2 loses contact. Reset KF.\n");
            if (current_train_index < MAX_TRAIN_SIZE && CF_count % 30 == 0) 
                {
                    // 保存状态输入特征 (statewhisker->KFoutput_1)
                    X_train[current_train_index * 2] = (234.0f - statewhisker->KFoutput_1) * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x * 1000;   // 第一个特征
                    X_train[current_train_index * 2 + 1] = - (statewhisker->p_y * 1000) - (234.0f - statewhisker->KFoutput_1) * sinf(statewhisker->yaw * (M_PI_F / 180.0f)); // 第二个特征

                    // 保存输出标签 y (设为 0)
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  // 更新训练数据索引
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            dis_net(statewhisker);
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 1 loses contact. Reset KF.\n");
            if (current_train_index < MAX_TRAIN_SIZE && CF_count % 30 == 0) 
                {
                    // 保存状态输入特征 (statewhisker->KFoutput_1)
                    X_train[current_train_index * 2] = (234.0f - statewhisker->KFoutput_2) * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x * 1000;   // 第一个特征
                    X_train[current_train_index * 2 + 1] = - (statewhisker->p_y * 1000) - (234.0f - statewhisker->KFoutput_2) * sinf(statewhisker->yaw * (M_PI_F / 180.0f)); // 第二个特征

                    // 保存输出标签 y (设为 0)
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  // 更新训练数据索引
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {   
            CF_count++;
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
            is_KF_initialized = 0;
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            DEBUG_PRINT("Exiting CF state, resources released.\n");
        }
        break;

    case backward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        backward_count--;
        if (backward_count < 0)
        {
            stateCF = transition(rotate);
            DEBUG_PRINT("Backward finished. Starting rotating.\n");
            rotate_count = 180;
        }
        break;

    case rotate:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        rotate_count--;
        if (rotate_count <0)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Rotate finished. Starting flying forward.\n");
            CF_count = 150;
            if (current_train_index < MAX_TRAIN_SIZE) 
                {
                    // 保存状态输入特征 (statewhisker->KFoutput_1)
                    X_train[current_train_index * 2] = statewhisker->p_x * 1000;   // 第一个特征
                    X_train[current_train_index * 2 + 1] = - (statewhisker->p_y * 1000); // 第二个特征

                    // 保存输出标签 y (设为 0)
                    y_train[current_train_index] = -1.0f;
                    current_train_index++;  // 更新训练数据索引
                    DEBUG_PRINT("Inside Training sample saved. Current index: %d\n", current_train_index);
                }
        }
        break;
    
    case GPIS:
        if (current_train_index > 0) 
        {
            // 调用 gp_fit 函数拟合模型，使用当前已收集的数据进行训练
            gp_fit(&gp_model, X_train, y_train, current_train_index);
            DEBUG_PRINT("GP model trained with %d training samples.\n", current_train_index);

            int grid_size = 10; // 网格的大小
            float x_min = -3.0f;
            float x_max = 3.0f;
            float y_min = -3.0f;
            float y_max = 3.0f;
            float x_step = (x_max - x_min) / (grid_size - 1);
            float y_step = (y_max - y_min) / (grid_size - 1);

            float X_test[2]; // 用于存储当前状态的输入
            float y_pred[1];  // 预测值
            float y_std[1];   // 预测标准

            // 遍历网格
            for (int i = 0; i < grid_size; ++i) 
            {
                for (int j = 0; j < grid_size; ++j) 
                {
                    // 生成网格点
                    X_test[0] = x_min + i * x_step;  // x 坐标
                    X_test[1] = y_min + j * y_step;  // y 坐标

                    // 调用 gp_predict 函数进行预测
                    gp_predict(&gp_model, X_test, 1, y_pred, y_std);  // 对每个网格点进行预测

                    // 输出预测结果
                    DEBUG_PRINT("GP prediction at (x = %f, y = %f): y_pred = %f, y_std = %f\n", 
                            X_test[0], X_test[1], y_pred[0], y_std[0]);
            }
        }
        break;
    }


    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    case backward:
        cmdVelXTemp = -1.0f * maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case rotate:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = -1.0f * maxTurnRate;
        break;

    case GPIS:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}