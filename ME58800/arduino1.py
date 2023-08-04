import serial
import redis
import re
import json

''' This program will get and extract message from uno1'''

usb_port = '/dev/ttyUSB0'
try: 
    serial_port = serial.Serial(usb_port, 115200)
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
        # print(ord(input_char))
        if input_char.decode("utf-8") != '\n':
            input_text.append(input_char.decode("utf-8"))
            # print(input_text)
        else:
            input_text.pop()
            message = "".join(input_text)
            # print("Read input " + message + " from Arduino")
            # re_pattern = r"(\w*):(-?)(\d*)"
            re_pattern = r"(\w*):(.*)"
            re_result = re.findall(re_pattern, message)[0]
            # print(re_result)
            if re_result[0] == 'Encoder':
                encoder_pattern = r"(-?)(\d*):(-?)(\d*)"
                encoder_result = re.findall(encoder_pattern, re_result[1])[0]
                # print(encoder_result)
                encoder_value['left'] = int(encoder_result[1])
                if encoder_result[0] == '-':
                    encoder_value['left'] *= -1
                encoder_value['right'] = int(encoder_result[3])
                if encoder_result[2] == '-':
                    encoder_value['right'] *= -1
                # print('Encoder : ', encoder_value)
                redisdb.set('encoder_value', json.dumps(encoder_value))

            elif re_result[0] == 'LineArray':
                line_value = re_result[1]
                # print(line_value)
                redisdb.set('line_value', line_value)

            elif re_result[0] == 'LeftIRArray':
                line_value = re_result[1]
                # print(line_value)
                redisdb.set('line_left', line_value)

            elif re_result[0] == 'FrontIR':
                front_ir = int(re_result[1])
                # print(front_ir)
                redisdb.set('front_ir', front_ir)

            elif re_result[0] == 'BallIR':
                ball_ir = int(re_result[1])
                
                if ball_ir < 60 and ball_ir > 10:
                    # print(ball_ir)
                    redisdb.set('ball_ir', ball_ir)

            input_text = []
    except Exception as e:
        print(e)
        exit(1)


