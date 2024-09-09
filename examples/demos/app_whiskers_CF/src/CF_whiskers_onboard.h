/*
 * wallfollowing_multirange_onboard.h
 *
 *  Created on: Aug 7, 2018
 *      Author: knmcguire
 */

#ifndef SRC_CF_WHISKERS_ONBOARD_H_
#define SRC_CF_WHISKERS_ONBOARD_H_
#include <stdint.h>
#include <stdbool.h>

typedef enum
{
    forward,
    hover,
    CF
} StateCF;

StateCF FSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1, float whisker2, float timeOuter);


void FSMInit(float MIN_THRESHOLD1_input, float MAX_THRESHOLD1_input, 
             float MIN_THRESHOLD2_input, float MAX_THRESHOLD2_input, 
             float maxSpeed_input, float maxTurnRate_input, StateCF initState);
#endif /* SRC_CF_WHISKERS_ONBOARD_H_ */
