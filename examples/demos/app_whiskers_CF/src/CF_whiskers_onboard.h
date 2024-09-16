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
    float zi[6][2];
    float time_stamp;
    float b[3], a[3];  // Filter coefficients (3-tap filter example)
} StateWhisker;

StateCF FSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, Whisker *whisker, float timeOuter);


void FSMInit(float MIN_THRESHOLD1_input, float MAX_THRESHOLD1_input, 
             float MIN_THRESHOLD2_input, float MAX_THRESHOLD2_input, 
             float maxSpeed_input, float maxTurnRate_input, StateCF initState);

void ProcessWhiskerInit(Whisker *whisker) {
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
};
#endif /* SRC_CF_WHISKERS_ONBOARD_H_ */
