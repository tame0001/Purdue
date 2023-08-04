from gpiozero import Robot, Motor
from simple_pid import PID
import json
import redis
import schedule

def speed_control():
    encoder = json.loads(redisdb.get('encoder_value').decode('ascii'))
    left_speed = abs(int(encoder['left']))
    right_speed = abs(int(encoder['right']))
    print(encoder)
    if state == 'STOP':
        motorL.stop()
        motorR.stop()
    elif state == 'FORWARD':
        left_pwm = left_PID(left_speed)
        right_pwm = right_PID(right_speed)
        motorL.forward(left_pwm)
        motorR.forward(right_pwm)
    elif state == 'BACKWARD':
        left_pwm = left_PID(left_speed)
        right_pwm = right_PID(right_speed)
        motorL.backward(left_pwm)
        motorR.backward(right_pwm)
    print(left_pwm, ' ', right_pwm)
    

redisdb = redis.Redis(host='localhost', port=6379, db=0)

# robot = Robot(left=('GPIO26', 'GPIO20'), right=('GPIO19', 'GPIO16'))
motorL = Motor('GPIO26', 'GPIO20')
motorR = Motor('GPIO19', 'GPIO16')

speed = 10

p = 2
i = 0.1
d = 0.05

left_PID = PID(p, i, d, setpoint=speed, output_limits=(0,1), sample_time=None)
right_PID = PID(p, i, d, setpoint=speed, output_limits=(0,1), sample_time=None)

state = 'FORWARD'

# schedule.every(1).seconds.do(speed_control)


while(1):
    # schedule.run_pending()
    speed_control()




