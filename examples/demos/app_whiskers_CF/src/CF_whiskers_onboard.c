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
// static MLPParams params_1; // 第一个 MLP 参数
// static MLPParams params_2; // 第二个 MLP 参数
static KalmanFilterWhisker kf1;
static KalmanFilterWhisker kf2;
static float process_noise = 0.1;
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
        if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
        }
        break;

    case backward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
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
