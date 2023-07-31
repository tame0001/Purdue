import serial
import re

port = input("Enter USB port: ")
usb_port =  port
print(usb_port)
try: 
    serial_port = serial.Serial(usb_port, 115200)
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
        # input_text.pop()
        message = "".join(input_text)
        print(message)
        input_text = []


