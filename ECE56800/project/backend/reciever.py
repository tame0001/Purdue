'''
    AUTHOR: Thirawat Bureetes
    EMAIL: tbureete@purdue.edu
    DETAIL: Thie program will get message from USB port and process incoming message into useful data.
            Then it will send data to internet via MQTT Protocol and save data in local Redis database.
'''
import serial
import re
import redis
import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # client.subscribe("$SYS/#")

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()
# client.loop_forever()

# port = input("Enter USB port: ")
# usb_port = "/dev/tty" + port
usb_port = '/dev/ttyACM0'
# print(usb_port)
try: 
    serial_port = serial.Serial(usb_port, 115200)
    redisdb = redis.Redis(host='localhost', port=6379, db=0)
except Exception as e:
    print(e)
    exit(1)

input_text = []
while 1:
    input_char = serial_port.read()
    # print(ord(input_char))
    if input_char.decode("utf-8") != '\n':
        input_text.append(input_char.decode("utf-8"))
        # print(input_text)
    else:
        input_text.pop()
        message = "".join(input_text)
        re_pattern = r"(\w*):(.*)"
        re_result = re.findall(re_pattern, message)
        # print(re_result)
        if len(re_result) > 0:
            # print(re_result)
            value = re_result[0][1]
            print(value)
            redisdb.set('sensor', value)
            client.publish('/ece568/sensor/raw', value)
            data_pattern = r"(\d*)"
            data_raw = re.findall(re_pattern, value)
            red = data_raw[0][0]
            remain = data_raw[0][1]
            redisdb.set('red', red)
            data_raw = re.findall(re_pattern, remain)
            green = data_raw[0][0]
            remain = data_raw[0][1]
            redisdb.set('green', green)
            data_raw = re.findall(re_pattern, remain)
            blue = data_raw[0][0]
            clear = data_raw[0][1]
            redisdb.set('blue', blue)
            redisdb.set('clear', clear)
            # print(clear)
            # print(data_raw[0][1])

            payload = {
                'red': red,
                'green': green,
                'blue': blue,
                'clear': clear
            }
            client.publish('/ece568/sensor/rgb', json.dumps(payload))
            
        # print(message)
        input_text = []


