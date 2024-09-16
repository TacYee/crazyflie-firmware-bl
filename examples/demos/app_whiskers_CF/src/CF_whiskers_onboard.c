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

typedef struct Whisker {
    float whisker1_1, whisker1_2, whisker1_3;
    float whisker2_1, whisker2_2, whisker2_3;
    float sum_x[6];
    float sum_y[6];
    float sum_x_squared[6];
    float sum_xy[6];
    float slopes[6];
    float intercepts[6];
    int count;
    float zi[6][2];
    float time_stamp;
    float b[3], a[3];  // Filter coefficients (3-tap filter example)
} Whisker;


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

void WhiskerInit(Whisker *whisker) {
    for (int i = 0; i < 6; i++) {
        whisker->zi[i][0] = 0.0f;  // Initialize filter state
        whisker->zi[i][1] = 0.0f;
    }
    
    whisker->count = 0;
    
    // 1st order butterworth filter 0.05-1hz
    whisker->b[0] = 0.05639124;
    whisker->b[1] = 0.0f;
    whisker->b[2] = -0.05639124;
    whisker->a[0] = 1.0f;
    whisker->a[1] = -1.88647164;
    whisker->a[2] = 0.88721752;
}

void update_statistics(Whisker *whisker, float data, int index) {
    float x = (float)whisker->count;
    whisker->sum_x[index] += x;
    whisker->sum_y[index] += data;
    whisker->sum_x_squared[index] += x * x;
    whisker->sum_xy[index] += x * data;
}


void calculate_parameters(Whisker *whisker) {
    int n = DATA_SIZE;
    for (int i = 0; i < 6; i++) {
        whisker->slopes[i] = (n * whisker->sum_xy[i] - whisker->sum_x[i] * whisker->sum_y[i]) /
                             (n * whisker->sum_x_squared[i] - whisker->sum_x[i] * whisker->sum_x[i]);
        whisker->intercepts[i] = (whisker->sum_y[i] - whisker->slopes[i] * whisker->sum_x[i]) / n;
    }
}


void apply_bandpass_filter(float data, float *zi, float *filtered_data, float *b, float *a) {
    float input = data;
    float output = b[0] * input + zi[0];
    zi[0] = b[1] * input - a[1] * output + zi[1];
    zi[1] = b[2] * input - a[2] * output;
    *filtered_data = output;
}



void process_data(Whisker *whisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) {
    float residuals[6];
    residuals[0] = whisker1_1 - (whisker->slopes[0] * whisker->time_stamp + whisker->intercepts[0]);
    residuals[1] = whisker1_2 - (whisker->slopes[1] * whisker->time_stamp + whisker->intercepts[1]);
    residuals[2] = whisker1_3 - (whisker->slopes[2] * whisker->time_stamp + whisker->intercepts[2]);
    residuals[3] = whisker2_1 - (whisker->slopes[3] * whisker->time_stamp + whisker->intercepts[3]);
    residuals[4] = whisker2_2 - (whisker->slopes[4] * whisker->time_stamp + whisker->intercepts[4]);
    residuals[5] = whisker2_3 - (whisker->slopes[5] * whisker->time_stamp + whisker->intercepts[5]);

    apply_bandpass_filter(residuals[0], whisker->zi[0], &whisker->whisker1_1, whisker->b, whisker->a);
    apply_bandpass_filter(residuals[1], whisker->zi[1], &whisker->whisker1_2, whisker->b, whisker->a);
    apply_bandpass_filter(residuals[2], whisker->zi[2], &whisker->whisker1_3, whisker->b, whisker->a);
    apply_bandpass_filter(residuals[3], whisker->zi[3], &whisker->whisker2_1, whisker->b, whisker->a);
    apply_bandpass_filter(residuals[4], whisker->zi[4], &whisker->whisker2_2, whisker->b, whisker->a);
    apply_bandpass_filter(residuals[5], whisker->zi[5], &whisker->whisker2_3, whisker->b, whisker->a);
    
    whisker->count++;  // Increment time stamp
}

void data_received(Whisker *whisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) {
    if (whisker->count < DATA_SIZE) {
        update_statistics(whisker, whisker1_1, 0);
        update_statistics(whisker, whisker1_2, 1);
        update_statistics(whisker, whisker1_3, 2);
        update_statistics(whisker, whisker2_1, 3);
        update_statistics(whisker, whisker2_2, 4);
        update_statistics(whisker, whisker2_3, 5);
        whisker->count++;
        
        if (whisker->count == DATA_SIZE) {
            calculate_parameters(whisker);
        }
    } else {
        process_data(whisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        whisker->count++;
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

StateCF FSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1, float whisker2, float timeOuter)
{

  timeNow = timeOuter;

  if (firstRun)
  {
    firstRun = false;
    StartTime = timeNow;
  }

  /***********************************************************
  * Handle state transitions
  ***********************************************************/
  switch (stateCF)
  {
  
  case hover:
    if (timeNow - StartTime >= waitForStartSeconds)
    {
      stateCF = transition(forward);
    }
    
  case forward:
    if (whisker1 > MIN_THRESHOLD1 || whisker2 > MIN_THRESHOLD2)
    {
      stateCF = transition(CF);
    }
    break;

  case CF:
    if (whisker1 < MIN_THRESHOLD1 && whisker2 < MIN_THRESHOLD2)
    {
      stateCF = transition(forward);
    }
    break;

  /***********************************************************
   * Handle state actions
   ***********************************************************/

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
    //fly side way
    if (MAX_THRESHOLD1 > whisker1 && whisker1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > whisker2 && whisker2 > MIN_THRESHOLD2)
    {
      cmdVelXTemp = 0.0f;
      cmdVelYTemp = -1.0f * maxSpeed;
      cmdAngWTemp = 0.0f;
    }

    //turn left
    else if (MAX_THRESHOLD1 > whisker1 && whisker1 > MIN_THRESHOLD1 && whisker2 < MIN_THRESHOLD2)
    {
      cmdVelXTemp = 0.0f;
      cmdVelYTemp = 0.0f;
      cmdAngWTemp = maxTurnRate;
    }

    else if (whisker1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > whisker2 && whisker2 > MIN_THRESHOLD2)
    {
      cmdVelXTemp = 0.0f;
      cmdVelYTemp = 0.0f;
      cmdAngWTemp = maxTurnRate;
    }
    
    //turn right
    else if (MAX_THRESHOLD1 > whisker1 && whisker1 > MIN_THRESHOLD1 && whisker2 > MAX_THRESHOLD2)
    {
      cmdVelXTemp = 0.0f;
      cmdVelYTemp = 0.0f;
      cmdAngWTemp = -1.0f * maxTurnRate;
    }

    else if (whisker1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > whisker2 && whisker2 > MIN_THRESHOLD2)
    {
      cmdVelXTemp = 0.0f;
      cmdVelYTemp = 0.0f;
      cmdAngWTemp = -1.0f * maxTurnRate;
    }

    //fly backward
    else
    {
      cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
      cmdVelYTemp = 0.0f;
      cmdAngWTemp = 0.0f;      
    }
    break;

  default:
    //State does not exist so hover!!
    cmdVelXTemp = 0.0f;
    cmdVelYTemp = 0.0f;
    cmdAngWTemp = 0.0f;
  }

  *cmdVelX = cmdVelXTemp;
  *cmdVelY = cmdVelYTemp;
  *cmdAngW = cmdAngWTemp;

  return stateCF;
}
