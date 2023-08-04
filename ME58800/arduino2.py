import serial
import redis
import re
import json

''' This program will get and extract message from uno2'''

usb_port = '/dev/ttyACM0'
try: 
    serial_port = serial.Serial(usb_port, 9600)
    redisdb = redis.Redis(host='localhost', port=6379, db=0)
except Exception as e:
    print(e)
    exit(1)

input_text = []
encoder_value = {
    'left': 0,
    'right': 0
}
while 1:
    try: 
        input_char = serial_port.read()
        if input_char.decode("utf-8") != '\n':
            input_text.append(input_char.decode("utf-8"))
            # print(input_text)
        else:
            if len(input_text) > 1:
                input_text.pop()
                message = "".join(input_text)
                # print("Read input " + message + " from Arduino")
                # re_pattern = r"(\w*):(-?)(\d*)"
                re_pattern = r"(\w*):(.*)"
                re_result = re.findall(re_pattern, message)[0]
                # print(re_result)
                if re_result[0] == 'Ball':
                    ball_state = re_result[1]
                    # print(ball_state)
                    redisdb.set('ball_state', ball_state)

            

            input_text = []
    except Exception as e:
        print(e)
        exit(1)


