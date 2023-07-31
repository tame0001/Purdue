/**
 * Copyright (c) 2015 - 2019, Nordic Semiconductor ASA
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form, except as embedded into a Nordic
 *    Semiconductor ASA integrated circuit in a product or a software update for
 *    such product, must reproduce the above copyright notice, this list of
 *    conditions and the following disclaimer in the documentation and/or other
 *    materials provided with the distribution.
 *
 * 3. Neither the name of Nordic Semiconductor ASA nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * 4. This software, with or without modification, must only be used with a
 *    Nordic Semiconductor ASA integrated circuit.
 *
 * 5. Any software provided in binary form under this license must not be reverse
 *    engineered, decompiled, modified and/or disassembled.
 *
 * THIS SOFTWARE IS PROVIDED BY NORDIC SEMICONDUCTOR ASA "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NORDIC SEMICONDUCTOR ASA OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
/** @file
 * @defgroup tw_sensor_example main.c
 * @{
 * @ingroup nrf_twi_example
 * @brief TWI Sensor Example main file.
 *
 * This file contains the source code for a sample application using TWI.
 *
 */

#include <stdio.h>
#include "boards.h"
#include "app_util_platform.h"
#include "app_error.h"
#include "nrf_drv_twi.h"
#include "nrf_delay.h"


#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"

/* TWI instance ID. */
#define TWI_INSTANCE_ID     0

/* Common addresses definition for temperature sensor. */
#define TCS34725_ADDR          (0x29U)

#define TCS34725_ID             (0x12U+(0x80U))
#define TCS34725_ATIME          (0x01U+(0x80U))
#define TCS34725_CONTROL        (0x0FU+(0x80U))
#define TCS34725_ENABLE         (0x00U+(0x80U))
#define TCS34725_CDATA          (0x14U+(0x80U))
#define TCS34725_CDATAH         (0x15U+(0x80U))
#define TCS34725_RDATA          (0x16U+(0x80U))
#define TCS34725_RDATAH         (0x17U+(0x80U))
#define TCS34725_GDATA          (0x18U+(0x80U))
#define TCS34725_GDATAH         (0x19U+(0x80U))
#define TCS34725_BDATA          (0x1AU+(0x80U))
#define TCS34725_BDATAH         (0x1BU+(0x80U))



/* Mode for LM75B. */
#define NORMAL_MODE 0U

/* Indicates if operation on TWI has ended. */
static volatile bool m_xfer_done = false;

/* TWI instance. */
static const nrf_drv_twi_t m_twi = NRF_DRV_TWI_INSTANCE(TWI_INSTANCE_ID);

/* Buffer for samples read from temperature sensor. */
static uint8_t m_sample[8];


/**
 * @brief Function for setting active mode on MMA7660 accelerometer.
 */
void TCS34725_init(void)
{
    ret_code_t err_code;

    uint8_t reg[2] = {TCS34725_ATIME, 0x00U};
    err_code = nrf_drv_twi_tx(&m_twi, TCS34725_ADDR, reg, sizeof(reg), false);
    APP_ERROR_CHECK(err_code);
    while (m_xfer_done == false);

    reg[0] = TCS34725_CONTROL;
    err_code = nrf_drv_twi_tx(&m_twi, TCS34725_ADDR, reg, sizeof(reg), false);
    APP_ERROR_CHECK(err_code);
    while (m_xfer_done == false);

    reg[0] = TCS34725_ENABLE;
    reg[1] = 0x01U;
//    err_code = nrf_drv_twi_tx(&m_twi, TCS34725_ADDR, reg, sizeof(reg), false);
//    APP_ERROR_CHECK(err_code);
//    while (m_xfer_done == false);

    NRF_LOG_INFO("Finished Initialize\n");

}

void TCS34725_enable(void)
{
    ret_code_t err_code;

    uint8_t reg[2] = {TCS34725_ENABLE, 0x01U};
    err_code = nrf_drv_twi_tx(&m_twi, TCS34725_ADDR, reg, sizeof(reg), false);
    APP_ERROR_CHECK(err_code);
    while (m_xfer_done == false);

    nrf_delay_ms(5);

    reg[1] = 0x03U;
    err_code = nrf_drv_twi_tx(&m_twi, TCS34725_ADDR, reg, sizeof(reg), false);
    APP_ERROR_CHECK(err_code);
    while (m_xfer_done == false);

    NRF_LOG_INFO("Enable TCS34725\n");

}


/**
 * @brief TWI events handler.
 */
void twi_handler(nrf_drv_twi_evt_t const * p_event, void * p_context)
{
    switch (p_event->type)
    {
        case NRF_DRV_TWI_EVT_DONE:
            if (p_event->xfer_desc.type == NRF_DRV_TWI_XFER_RX)
            {
//                data_handler(m_sample);
                  for(uint8_t i=0; i < sizeof(m_sample); i++){
                    NRF_LOG_INFO("Value: %d.", m_sample[i]);
                   }
                  NRF_LOG_INFO("Clear: %d.", m_sample[0] + (m_sample[1]<<8));
                  NRF_LOG_INFO("Red: %d.", m_sample[2] + (m_sample[3]<<8));
                  NRF_LOG_INFO("Green: %d.", m_sample[4] + (m_sample[5]<<8));
                  NRF_LOG_INFO("Blue: %d.", m_sample[6] + (m_sample[7]<<8));
            }
            m_xfer_done = true;
            break;
        default:
            break;
    }
}

/**
 * @brief UART initialization.
 */
void twi_init (void)
{
    ret_code_t err_code;

    const nrf_drv_twi_config_t twi_lm75b_config = {
       .scl                = ARDUINO_SCL_PIN,
       .sda                = ARDUINO_SDA_PIN,
       .frequency          = NRF_DRV_TWI_FREQ_100K,
       .interrupt_priority = APP_IRQ_PRIORITY_HIGH,
       .clear_bus_init     = false
    };

    err_code = nrf_drv_twi_init(&m_twi, &twi_lm75b_config, twi_handler, NULL);
    APP_ERROR_CHECK(err_code);

    nrf_drv_twi_enable(&m_twi);
}

/**
 * @brief Function for reading data from temperature sensor.
 */
static void read_sensor_data()
{
    m_xfer_done = false;

    uint8_t reg[2];
    reg[0] = TCS34725_CDATA;
//    NRF_LOG_INFO("%X\n", TCS34725_RDATAH);
    m_xfer_done = false;
    ret_code_t err_code = nrf_drv_twi_tx(&m_twi, TCS34725_ADDR, reg, 1, false);
    APP_ERROR_CHECK(err_code);
    while (m_xfer_done == false);

    /* Read 1 byte from the specified address - skip 3 bits dedicated for fractional part of temperature. */
    err_code = nrf_drv_twi_rx(&m_twi, TCS34725_ADDR, &m_sample, sizeof(m_sample));
    APP_ERROR_CHECK(err_code);
}

/**
 * @brief Function for main application entry.
 */
int main(void)
{
    APP_ERROR_CHECK(NRF_LOG_INIT(NULL));
    NRF_LOG_DEFAULT_BACKENDS_INIT();

    NRF_LOG_INFO("\r\nTWI sensor example started.");
    NRF_LOG_FLUSH();
    twi_init();
    TCS34725_init();
    NRF_LOG_FLUSH();
    TCS34725_enable();
    NRF_LOG_FLUSH();

    while (true)
    {
        nrf_delay_ms(1000);
        
        do
        {
            __WFE();
        }while (m_xfer_done == false);

        read_sensor_data();
        NRF_LOG_FLUSH();
    }
}

/** @} */
