/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2019 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * push.c - App layer application of the onboard push demo. The crazyflie 
 * has to have the multiranger and the flowdeck version 2.
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "commander.h"

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"

#include "log.h"
#include "param.h"
#include <math.h>
#include "usec_time.h"

#include "CF_whiskers_onboard.h"

#define DEBUG_MODULE "CF"

static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;


  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;


  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;

  setpoint->velocity_body = true;
}


// States
typedef enum
{
  idle,
  lowUnlock,
  unlocked,
  stopping
} StateOuterLoop;

StateOuterLoop stateOuterLoop = idle;
StateCF stateInnerLoop = hover;

// Some wallfollowing parameters and logging
float MIN_THRESHOLD1 = 20.0f;
float MAX_THRESHOLD1 = 100.0f;
float MIN_THRESHOLD2 = 20.0f;
float MAX_THRESHOLD2 = 100.0f;
float maxSpeed = 0.2f;
float maxTurnRate = 25.0f;

float cmdVelX = 0.0f;
float cmdVelY = 0.0f;
float cmdAngWRad = 0.0f;
float cmdAngWDeg = 0.0f;

#define MAX(a,b) ((a>b)?a:b)
#define MIN(a,b) ((a<b)?a:b)

void appMain()
{

  vTaskDelay(M2T(3000));
  // Getting Logging IDs of the whiskers
  logVarId_t idwhisker1_1 = logGetVarId("Whisker", "Barometer1_1");
  logVarId_t idwhisker1_2 = logGetVarId("Whisker", "Barometer1_2");
  logVarId_t idwhisker1_3 = logGetVarId("Whisker", "Barometer1_3");
  logVarId_t idwhisker2_1 = logGetVarId("Whisker1", "Barometer2_1");
  logVarId_t idwhisker2_2 = logGetVarId("Whisker1", "Barometer2_2");
  logVarId_t idwhisker2_3 = logGetVarId("Whisker1", "Barometer2_3");
  logVarId_t idFront = logGetVarId("range", "front");
  
  paramVarId_t idPositioningDeck = paramGetVarId("deck", "bcFlow2");
  paramVarId_t idMultiranger = paramGetVarId("deck", "bcMultiranger");

  // Initialize the wall follower state machine
  FSMInit(MIN_THRESHOLD1, MAX_THRESHOLD1, MIN_THRESHOLD2, MAX_THRESHOLD2, maxSpeed, maxTurnRate, stateInnerLoop);

  // Intialize the setpoint structure
  setpoint_t setpoint;

  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) {
    vTaskDelay(M2T(20));
    //DEBUG_PRINT(".");

    uint8_t positioningInit = paramGetUint(idPositioningDeck);
    uint8_t multirangerInit = paramGetUint(idMultiranger);

    if (stateOuterLoop == unlocked) {

      float cmdHeight = 0.3f;
      cmdVelX = 0.0f;
      cmdVelY = 0.0f;
      cmdAngWDeg = 0.0f;

      float whisker1_1 = logGetUint(idwhisker1_1);
      float whisker1_2 = logGetUint(idwhisker1_2);
      float whisker1_3 = logGetUint(idwhisker1_3);
      float whisker2_1 = logGetUint(idwhisker2_1);
      float whisker2_2 = logGetUint(idwhisker2_2);
      float whisker2_3 = logGetUint(idwhisker2_3);

      // The wall-following state machine which outputs velocity commands
      float timeNow = usecTimestamp() / 1e6;
      stateInnerLoop = FSM(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker2_1, timeNow);

      if (1) {
        setHoverSetpoint(&setpoint, cmdVelX, cmdVelY, cmdHeight, cmdAngWDeg);
        commanderSetSetpoint(&setpoint, 3);
      }

    }
  }
}
