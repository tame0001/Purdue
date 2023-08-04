from gpiozero import AngularServo

servo = AngularServo(17)
# servo.angle = 0
while True:
    servo.angle = 0
    print(servo.angle)