/*
 * wall_follower_multi_ranger_onboard.c
 *
 *  Created on: Sep 9, 2024
 *      Author: Chaoxiang Ye

 */

#include "CF_whiskers_onboard.h"
#include <math.h>
#include <stdlib.h>

#define DATA_SIZE 100

static float firstRun = false;
static float TurnRate = 0.5f;
static float maxSpeed = 0.2f;
static float maxTurnRate = 25.0f;
static float MIN_THRESHOLD1 = 20.0f;
static float MAX_THRESHOLD1 = 100.0f;
static float MIN_THRESHOLD2 = 20.0f;
static float MAX_THRESHOLD2 = 100.0f;
static float StartTime;
static const float waitForStartSeconds = 3.0f;
static float stateStartTime;

static StateCF stateCF = hover;
float timeNow = 0.0f;

void ProcessWhiskerInit(StateWhisker *statewhisker) {
    for (int i = 0; i < 6; i++) {
        statewhisker->zi[i][0] = 0.0f;  // Initialize filter state
        statewhisker->zi[i][1] = 0.0f;
    }
    
    statewhisker->count = 0;
    
    // 1st order butterworth filter 0.05-1hz
    statewhisker->b[0] = 0.05639124;
    statewhisker->b[1] = 0.0f;
    statewhisker->b[2] = -0.05639124;
    statewhisker->a[0] = 1.0f;
    statewhisker->a[1] = -1.88647164;
    statewhisker->a[2] = 0.88721752;
}

void update_statistics(StateWhisker *statewhisker, float data, int index) {
    float x = (float)statewhisker->count;
    statewhisker->sum_x[index] += x;
    statewhisker->sum_y[index] += data;
    statewhisker->sum_x_squared[index] += x * x;
    statewhisker->sum_xy[index] += x * data;
}


void calculate_parameters(StateWhisker *statewhisker) {
    int n = DATA_SIZE;
    for (int i = 0; i < 6; i++) {
        statewhisker->slopes[i] = (n * statewhisker->sum_xy[i] - statewhisker->sum_x[i] * statewhisker->sum_y[i]) /
                             (n * statewhisker->sum_x_squared[i] - statewhisker->sum_x[i] * statewhisker->sum_x[i]);
        statewhisker->intercepts[i] = (statewhisker->sum_y[i] - statewhisker->slopes[i] * statewhisker->sum_x[i]) / n;
    }
}


void apply_bandpass_filter(float data, float *zi, float *filtered_data, float *b, float *a) {
    float input = data;
    float output = b[0] * input + zi[0];
    zi[0] = b[1] * input - a[1] * output + zi[1];
    zi[1] = b[2] * input - a[2] * output;
    *filtered_data = output;
}



void process_data(StateWhisker *statewhisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) {
    float residuals[6];
    residuals[0] = whisker1_1 - (statewhisker->slopes[0] * statewhisker->time_stamp + statewhisker->intercepts[0]);
    residuals[1] = whisker1_2 - (statewhisker->slopes[1] * statewhisker->time_stamp + statewhisker->intercepts[1]);
    residuals[2] = whisker1_3 - (statewhisker->slopes[2] * statewhisker->time_stamp + statewhisker->intercepts[2]);
    residuals[3] = whisker2_1 - (statewhisker->slopes[3] * statewhisker->time_stamp + statewhisker->intercepts[3]);
    residuals[4] = whisker2_2 - (statewhisker->slopes[4] * statewhisker->time_stamp + statewhisker->intercepts[4]);
    residuals[5] = whisker2_3 - (statewhisker->slopes[5] * statewhisker->time_stamp + statewhisker->intercepts[5]);

    apply_bandpass_filter(residuals[0], statewhisker->zi[0], &statewhisker->whisker1_1, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[1], statewhisker->zi[1], &statewhisker->whisker1_2, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[2], statewhisker->zi[2], &statewhisker->whisker1_3, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[3], statewhisker->zi[3], &statewhisker->whisker2_1, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[4], statewhisker->zi[4], &statewhisker->whisker2_2, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[5], statewhisker->zi[5], &statewhisker->whisker2_3, statewhisker->b, statewhisker->a);
}

void ProcessDataReceived(StateWhisker *statewhisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) {
    if (statewhisker->count < DATA_SIZE) {
        update_statistics(statewhisker, whisker1_1, 0);
        update_statistics(statewhisker, whisker1_2, 1);
        update_statistics(statewhisker, whisker1_3, 2);
        update_statistics(statewhisker, whisker2_1, 3);
        update_statistics(statewhisker, whisker2_2, 4);
        update_statistics(statewhisker, whisker2_3, 5);
        statewhisker->count++;
        
        if (statewhisker->count == DATA_SIZE) {
            calculate_parameters(statewhisker);
        }
    } else {
        process_data(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        statewhisker->count++;
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
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > MIN_THRESHOLD1 || statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            stateCF = transition(CF);
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 < MIN_THRESHOLD1 && statewhisker->whisker2_1 < MIN_THRESHOLD2)
        {
            stateCF = transition(forward);
        }
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
        if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
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
