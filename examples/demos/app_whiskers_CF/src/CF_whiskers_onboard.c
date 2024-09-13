/*
 * wall_follower_multi_ranger_onboard.c
 *
 *  Created on: Sep 9, 2024
 *      Author: Chaoxiang Ye

 */

#include "CF_whiskers_onboard.h"
#include <math.h>

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
