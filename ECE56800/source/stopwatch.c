/*
 * Copyright (c) 2014, Freescale Semiconductor, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * o Redistributions of source code must retain the above copyright notice, this list
 *   of conditions and the following disclaimer.
 *
 * o Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or
 *   other materials provided with the distribution.
 *
 * o Neither the name of Freescale Semiconductor, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "MKL46Z4.h"
#include "slcd.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// -----------------------------------------------------------------
// Place your macro here - Start

#define SW1 (3)
#define SW3 (12)

#define PIN(x) (1 << x)

#define GREEN_LED (5)
#define GREEN_LED_ON() PTD->PCOR |= PIN(GREEN_LED) ;
#define GREEN_LED_OFF() PTD->PSOR |= PIN(GREEN_LED) ;
#define GREEN_LED_TOGGLE()   PTD->PTOR |= PIN(GREEN_LED) ;

#define RED_LED (29)
#define RED_LED_ON() PTE->PCOR |= PIN(RED_LED) ;
#define RED_LED_OFF() PTE->PSOR |= PIN(RED_LED) ;
#define RED_LED_TOGGLE()   PTE->PTOR |= PIN(RED_LED) ;


// Place your macro here - End
// -----------------------------------------------------------------

typedef enum {
	STOP,
	RUN,
	PAUSED
} enumStopWatchOperationState;

enumStopWatchOperationState enumStopWatchState = STOP;

bool bStartStopSwitchPressed = false;
bool bResetSwitchPressed     = false;

bool bIsTimerExpired         = false;

unsigned char    ucSecond = 0;
unsigned char    ucHundredsMilliSecond = 0;
unsigned char    ucMinute = 0;
unsigned short   usTimeElapsed = 0;

unsigned char    ucaryLCDMsg[5] = "";

// --------------------------------------------------------------------
// Place your global variable(s) here - Start

uint32_t sw1_status, sw3_status;

// Place your global variable(s) here - End
// --------------------------------------------------------------------

void LED_Init(void)
{
// --------------------------------------------------------------------
// Place your LED related code and register settings here - Start

	PORTD->PCR[GREEN_LED] = PORT_PCR_MUX(1);
	PTD->PDDR |= PIN(GREEN_LED);
	PTD->PSOR |= PIN(GREEN_LED);

	PORTE->PCR[RED_LED] = PORT_PCR_MUX(1);
	PTE->PDDR |= PIN(RED_LED);
	PTE->PSOR |= PIN(RED_LED);

// Place your LED related code and register settings here - End
// --------------------------------------------------------------------
}

void SWITCH_Init(void)
{
// --------------------------------------------------------------------
// Place your switch related code and register settings here - Start

	PORTC->PCR[SW1] |= PORT_PCR_PE_MASK | PORT_PCR_PS_MASK | PORT_PCR_MUX(1) | PORT_PCR_IRQC(10);

	PORTC->PCR[SW3] |= PORT_PCR_PE_MASK | PORT_PCR_PS_MASK | PORT_PCR_MUX(1) | PORT_PCR_IRQC(10);

// Place your switch related code and register settings here - End
// --------------------------------------------------------------------
}

void TIMER_Init(void)
{
// --------------------------------------------------------------------
// Place your timer related code and register settings here - Start

	SIM->SCGC6 |= SIM_SCGC6_TPM0_MASK;
	SIM->SOPT2 |= SIM_SOPT2_TPMSRC(2);
	TPM0->MOD = TPM_MOD_MOD(24999); // 8Mhz / pre-scale 32 = 250,000 counter per second = 25,000 per .1 second
	TPM0->SC |= TPM_SC_PS(5) | TPM_SC_CMOD(1) | TPM_SC_TOIE_MASK;



// Place your timer related code and register settings here - End
// --------------------------------------------------------------------
}

void TPM0_IRQHandler(void)
{
// --------------------------------------------------------------------
// Place your timer ISR related code and register settings here - Start

	TPM0->SC |= TPM_SC_TOF_MASK; // Clear flag
	bIsTimerExpired = true;


// Place your timer ISR related code and register settings here - End
// --------------------------------------------------------------------
}

void PORTC_PORTD_IRQHandler(void)
{
// --------------------------------------------------------------------
// Place your port ISR related code and register settings here - Start

	PORTC->PCR[SW1] |= PORT_PCR_ISF_MASK;
	PORTC->PCR[SW3] |= PORT_PCR_ISF_MASK;
	sw1_status = ((PTC->PDIR)&PIN(SW1))>>SW1;
	sw3_status = ((PTC->PDIR)&PIN(SW3))>>SW3;
	if(sw3_status == 0){
		bStartStopSwitchPressed = true;
	}
	if(sw1_status == 0){
		bResetSwitchPressed = true;
		}


// Place your port ISR related code and register settings here - End
// --------------------------------------------------------------------
}

void main(void)
{
// --------------------------------------------------------------------
// Place your local variable(s) here - Start

// Place your local variable(s) here - End
// --------------------------------------------------------------------

    /*========================================================*/
    /*========================================================*/
    /*   Initialization                                       */
    /*========================================================*/
    /* Disable global interrupt */
    __disable_irq();

    /* Peripheral initialization */
    SLCD_Init();
    LED_Init();
    SWITCH_Init();
    TIMER_Init();

    /* Enable individual interrupt */
    NVIC_EnableIRQ(PORTC_PORTD_IRQn);
    NVIC_EnableIRQ(TPM0_IRQn);

    /* Enable global interrupt */
    __enable_irq();

    /*========================================================*/
    while(1){
// The codes between the '#if 0' to corresponding '#endif' are initially commented out.
// Right after you import the project, try to build it to confirm the successful project import.
// Once you confirm the successful compilation, remove the '#if 0' and the corresponding '#endif' directives before you start implementing your own codes.
#if 1
      /* State transition upon a switch-press */
      // Check if SW3 is pressed
        if(bStartStopSwitchPressed == true){
          // Clear the flag
            bStartStopSwitchPressed = false;
            if(enumStopWatchState == STOP){
                // Turn off the red LED
                RED_LED_OFF();
                enumStopWatchState = RUN;
            }else if(enumStopWatchState == RUN){
                enumStopWatchState = PAUSED;
            }else if(enumStopWatchState == PAUSED){
                RED_LED_OFF();
                enumStopWatchState = RUN;
            }
        // Check if SW1 is pressed
        }else if(bResetSwitchPressed == true){
          // Clear the flag
            bResetSwitchPressed = false;
            if(enumStopWatchState == STOP){
                // Nothing to be done
            }else if(enumStopWatchState == RUN){
                // Nothing to be done
            }else if(enumStopWatchState == PAUSED){
                enumStopWatchState = STOP;
            }
        }
        /* Carry out the given tasks defined in the current state */
        if(enumStopWatchState == STOP){
            // (Re)initialize variables
            ucSecond = 0;
            ucHundredsMilliSecond = 0;
            ucMinute = 0;
            usTimeElapsed = 0;
            // Write a message on the LCD
            SLCD_WriteMsg((unsigned char *)"STOP");
            // The red LED is turned on while the stopwatch is in standby
            RED_LED_ON();
        }else if(enumStopWatchState == RUN){
            // Check if timer is expired
            if(bIsTimerExpired == true){
                // Clear the flag
                bIsTimerExpired = false;
                // Increment the variable that takes care of hundreds-milliseconds
                ucHundredsMilliSecond++;
            }
            //  10 * 100 ms = 1 s
            if(ucHundredsMilliSecond == 10){
                // The variable for hundreds-milliseconds rolls over back to zero
                ucHundredsMilliSecond = 0;
                // Increment the variable that takes care of seconds
                ucSecond++;
                // Toggling the green LED every second
                GREEN_LED_TOGGLE();
            }
            // 1 min = 60 s
            if(ucSecond == 60){
                // The variable for seconds rolls over back to zero
                ucSecond = 0;
                // Increment the variable that takes care of minutes
                ucMinute++;
            }
            // 10 * 1 min = 10 min
            if(ucMinute == 10){
                // The variable for minutes rolls over back to zero due to limited digits in the LCD
                ucMinute = 0;
            }
            // The red LED is turned off indicating stopwatch is in operation
            RED_LED_OFF();
            // Combine the time-related subcomponents for LCD visualization
            usTimeElapsed = ucMinute * 1000 + ucSecond * 10 + ucHundredsMilliSecond;
            // Convert integer to string
            snprintf(ucaryLCDMsg, 5,"%4d",usTimeElapsed);
            // Write elapsed time on the LCD
            SLCD_WriteMsg(ucaryLCDMsg);
        }else if(enumStopWatchState == PAUSED){
            // Make sure the green LED is turned off
            GREEN_LED_OFF();
            // Check if timer is expired
            if(bIsTimerExpired == true){
                // Clear the flag
                bIsTimerExpired = false;
                // Toggling the red LED to indicate the stopwatch is paused
                RED_LED_TOGGLE();
            }
        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
// EOF
////////////////////////////////////////////////////////////////////////////////
