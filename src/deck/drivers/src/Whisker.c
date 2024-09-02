#define DEBUG_MODULE "WHISKER"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "debug.h"
#include "log.h"
#include "deck.h"
#include "uart1.h"
#include "FreeRTOS.h"
#include "task.h"
#include "system.h"

#define MAX_MESSAGE_SIZE 48

static uint8_t isInit = 0;
// uint32_t loggedTimestamp = 0;
float barometer1_1, barometer1_2, barometer1_3;

void readSerial1() {
    char buf[MAX_MESSAGE_SIZE];
    char c = '0';
    char* token = NULL;
    int tokenCount = 0;

    for (int i = 0; i < MAX_MESSAGE_SIZE - 1; i++) {
        if (!uart1GetDataWithDefaultTimeout(&c)) break;

        if (c == '\n') {
            buf[i] = '\0';

            // 使用strtok分割字符串
            token = strtok(buf, ",");
            while (token != NULL && tokenCount < 3) {
                switch (tokenCount) {
                    case 0:
                        barometer1_1 = atof(token);
                        break;
                    case 1:
                        barometer1_2 = atof(token);
                        break;
                    case 2:
                        barometer1_3 = atof(token);
                        break;
                }

                token = strtok(NULL, ",");
                tokenCount++;
            }

            if (tokenCount < 3) {
                DEBUG_PRINT("Error parsing sensor1 data\n");
            }

            break;
        }

        buf[i] = c;

        if (c == ',') {
            // 重置 token 和 tokenCount
            token = NULL;
            tokenCount = 0;
        }
    }
}



void WhiskerTask(void *param) {
    systemWaitStart();
    while (1) {
        readSerial1();
    }
}

static void WhiskerInit() {
    DEBUG_PRINT("Initialize driver\n");

    uart1Init(115200);

    xTaskCreate(WhiskerTask, WHISKER_TASK_NAME, WHISKER_TASK_STACKSIZE, NULL,
                WHISKER_TASK_PRI, NULL);

    isInit = 1;
}

static bool WhiskerTest() {
    return isInit;
}

static const DeckDriver WhiskerDriver = {
        .name = "Whisker",
        .init = WhiskerInit,
        .test = WhiskerTest,
        .usedPeriph = DECK_USING_UART1,
};

DECK_DRIVER(WhiskerDriver);


/**
 * Logging variables for the Whisker
 */
LOG_GROUP_START(Whisker)
LOG_ADD(LOG_FLOAT, Barometer1_1, &barometer1_1)
LOG_ADD(LOG_FLOAT, Barometer1_2, &barometer1_2)
LOG_ADD(LOG_FLOAT, Barometer1_3, &barometer1_3)
LOG_GROUP_STOP(Whisker) 