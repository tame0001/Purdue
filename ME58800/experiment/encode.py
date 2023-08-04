import RPi.GPIO as GPIO
import time

def channelA_event_handler(pin):
    print("detect edge")

channalA = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(channalA, GPIO.IN)
GPIO.add_event_detect(channalA, GPIO.RISING)
GPIO.add_event_callback(channalA, channelA_event_handler)

while True:
    time.sleep(10)
    print("ping")