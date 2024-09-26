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
    CF,
    backward
} StateCF;


typedef struct{
    float whisker1_1, whisker1_2, whisker1_3;
    float whisker2_1, whisker2_2, whisker2_3;
    float sum_x[6];
    float sum_y[6];
    float sum_x_squared[6];
    float sum_xy[6];
    float slopes[6];
    float intercepts[6];
    int count;
    int preprocesscount;
    float zi[6][2];
    float b[3], a[3];
    float mlpoutput_1;
    float mlpoutput_2;
    float KFoutput_1;
    float KFoutput_2;
} StateWhisker;

StateCF FSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter);
StateCF MLPFSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter);
StateCF FSM_data(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter);


void FSMInit(float MIN_THRESHOLD1_input, float MAX_THRESHOLD1_input, 
             float MIN_THRESHOLD2_input, float MAX_THRESHOLD2_input, 
             float maxSpeed_input, float maxTurnRate_input, StateCF initState);

void ProcessWhiskerInit(StateWhisker *statewhisker);
#endif /* SRC_CF_WHISKERS_ONBOARD_H_ */
