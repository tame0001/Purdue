#include "MKL46Z4.h"

#define PIN(x) (1 << x)

// Green LED is connected to PTD5

#define GREEN_LED (5)
#define GREEN_LED_ON() PTD->PCOR |= PIN(GREEN_LED) ;
#define GREEN_LED_OFF() PTD->PSOR |= PIN(GREEN_LED) ;
#define GREEN_LED_TOGGLE()   PTD->PTOR |= PIN(GREEN_LED) ;

// Red LED is connected to PTE29

#define RED_LED (29)
#define RED_LED_ON() PTE->PCOR |= PIN(RED_LED) ;
#define RED_LED_OFF() PTE->PSOR |= PIN(RED_LED) ;
#define RED_LED_TOGGLE()   PTE->PTOR |= PIN(RED_LED) ;

// SW1 is connected to PTC3

#define SW1 (3)

// SW3 is connected to PTC12

#define SW3 (12)

void delay ( unsigned int uiDelayCycles ) {

	for (int i = 0 ; i < uiDelayCycles ; i++);

}

void main ( void ) {

	uint32_t sw1_status, sw3_status;

	/*========================================================*/
	/*Green LED 											  */
	/*========================================================*/

	/*Turn on  clock  to PortD module*/
	/*(KL46 Sub-Family Reference Manual (p. 209))*/

	SIM->SCGC5 |= SIM_SCGC5_PORTD_MASK;

	/*Set the PTD5 pin multiplexer to GPIO mode*/
	/*(KL46 Sub-Family  Reference  Manual (p. 193))*/

	PORTD->PCR[GREEN_LED] = PORT_PCR_MUX(1);

	/*Set the pins direction to output*/
	/*(KL46 Sub-Family  Reference  Manual (p. 838))*/

	PTD->PDDR |= PIN(GREEN_LED);

	/*Set the initial output state to low*/
	/*(KL46 Sub-Family  Reference  Manual (p. 836))*/

	PTD->PSOR |= PIN(GREEN_LED);

	/*Turn on  clock  to PortD module*/

	SIM->SCGC5 |= SIM_SCGC5_PORTE_MASK;

	/*Set the PTE29 pin multiplexer to GPIO mode*/

	PORTE->PCR[RED_LED] = PORT_PCR_MUX(1);

	/*Set the pins direction to output*/

	PTE->PDDR |= PIN(RED_LED);

	/*Set the initial output state to low*/

	PTE->PSOR |= PIN(RED_LED);

	/*Turn on  clock  to PortC module*/

	SIM->SCGC5 |= SIM_SCGC5_PORTC_MASK;

	/*Set the PTC3 pin multiplexer to GPIO mode and enable internal pull-up*/

	PORTC->PCR[SW1] = PORT_PCR_MUX(1);
	PORTC->PCR[SW1] |= PORT_PCR_PE_MASK;
	PORTC->PCR[SW1] |= PORT_PCR_PS_MASK;

	/*Set the PTC3 pin multiplexer to GPIO mode and enable internal pull-up*/

	PORTC->PCR[SW3] = PORT_PCR_MUX(1);
	PORTC->PCR[SW3] |= PORT_PCR_PE_MASK;
	PORTC->PCR[SW3] |= PORT_PCR_PS_MASK;


	/*========================================================*/

	while (1) {

		sw1_status = ((PTC->PDIR)&PIN(SW1))>>SW1;
		sw3_status = ((PTC->PDIR)&PIN(SW3))>>SW3;

		if(sw1_status == 0 && sw3_status == 0){
			PTE->PSOR |= PIN(RED_LED);
			PTD->PSOR |= PIN(GREEN_LED);
		}
		else if(sw1_status == 1 && sw3_status == 0){
			PTE->PCOR |= PIN(RED_LED);
			PTD->PSOR |= PIN(GREEN_LED);
		}
		else if(sw1_status == 0 && sw3_status == 1){
			PTE->PSOR |= PIN(RED_LED);
			PTD->PCOR |= PIN(GREEN_LED);
		}
		else if(sw1_status == 1 && sw3_status == 1){
			PTE->PSOR |= PIN(RED_LED);
			PTD->PSOR |= PIN(GREEN_LED);
		}



//		delay (4000000) ; // roughly one second

	}

}
