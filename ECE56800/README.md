#### AUTHOR: Thirawat Bureetes
#### EMAIL: tbureete@purdue.edu

This system consist of four devices s
1. PC as end user device
2. Raspberry Pi as local cloud sever
3. nRF Development Kit as BLE central
4. nRF Development Kit as BLE peripheral

The system provides two options for end user
1. Real-Time web application
2. REST API 

There are several options to host webserver. This npm package called "http-server". 
It is available for both Window and Linux. 
Installation methold could be found here: https://www.npmjs.com/package/http-server
To install npm, please follow: https://www.npmjs.com/get-npm
To use it, go to "fronted" folder and activate "http-server". 
This operation can be used in either PC or Raspberry Pi. 
To open the website, browse localhost in case of running "http-server" in PC.
Or Raspberry Pi IP-address in case of running "http-server" in Raspberry Pi.
Be sure that device connect to internet and the real-time information will delivered by MQTT protocol from public MQTT broker.

The REST API are powered by Python Flask framework. It is recommended to run it on Raspberry Pi.
The required packages are listed in "requirements.txt". To activate the REST API server, only "app.py" needs to be run.
As data come to Raspberry Pi via USB port, the "receiver.py" will deal with incomming data. 
This file will need root priviledge to access USB port. Moreever, Redis database will be use to store data.
To install Redis, please follow: https://redis.io/ or install directly by Linux repository.
Both "receiver.py" and "app.py" need to be run together.

Nodic provides Software Development Kit (SDK), please download: https://www.nordicsemi.com/Software-and-Tools/Software/nRF5-SDK
The version of SDK in used is 15.3.0. To upload firmware for BLE central node, copy "main.c" from "ble/central" to SDK->example->ble_central->ble_app_uart_c
For central node, there is no other file required. The Development Kit can be flashed directly by USB cable. No external programmer required.
Ther are two options to do:
1. With SEGGER Embedded Studio: go to pca10040->s132->ses 
2. With Keil: go to pca10040->s132->arm5_no_packs
then run "ble_app_uart_c_pca10040_s132. The project will be opened with selected IDE. 
Compile then upload.

For peripheral node, copy "main.py" from "ble/peripheral" to SDK->example->ble_peripheral->ble_app_uart
Open the IDE the same way as central node. Unlike central, there are some files that need to be imported manually.
1. nrfx_prs.c 
2. nrfx_prs.h 
3. nrfx_twim.c
All files can be found in SDK folder or also provided in "ble/peripheral".
In IDE, open "sdk_config.h" search for NRFX_TWIM_ENABLE and NRFX_TWI_ENABLE. Change to value to 1.
Compile then upload.