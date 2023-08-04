from gpiozero import Motor
from gpiozero import LED
import redis
import json
import time

drop = LED(17)
pinA = LED(27)
pinB = LED(22)

class Chasis(object):
    ''' this class will control the movement of the robot'''

    def __init__(self, leftF='GPIO13', leftB='GPIO20', rightF='GPIO19', rightB='GPIO16'):
        self.left_motor = Motor(leftF, leftB)
        self.right_motor = Motor(rightF, rightB)
        self.driving_speed = 0.5
        self.turning_speed = 0.4
        self.sliding_speed = 0.1
        self.acceleration_rate = 0.05
        self.state = 'STOP'
        self.redisdb = redis.Redis(host='localhost', port=6379, db=0)

    def drive_forward(self):
        if self.state != 'FORWARD':
            current_speed = 0
            while current_speed < self.driving_speed-0.2:
                current_speed += self.acceleration_rate
                self.left_motor.forward(current_speed+0.3 if current_speed < 0.5 else 1)
                self.right_motor.forward(current_speed) 
                time.sleep(0.05)
               
        self.left_motor.forward(self.driving_speed)
        self.right_motor.forward(self.driving_speed)
        self.state = 'FORWARD'
    
    def drive_backward(self):
        self.left_motor.backward(self.driving_speed)
        self.right_motor.backward(self.driving_speed)
        self.state = 'BACKWARD'

    def stop(self):
        if self.state != 'STOP':
            current_speed = self.driving_speed
            while current_speed < self.driving_speed:
                current_speed -= self.acceleration_rate
                self.left_motor.forward(current_speed)
                self.right_motor.forward(current_speed) 
                time.sleep(0.05)
        self.right_motor.stop()
        time.sleep(0.05)
        self.left_motor.stop()
        self.state = 'STOP'

    def turn_right(self, degree_to_turn):
        initial_encoder_value = self.get_current_encoder()
        while self.calculate_degree(initial_encoder_value) < degree_to_turn:
            self.right_motor.backward(self.turning_speed)
            self.left_motor.stop()
        
        if self.state == 'FORWARD':
            self.drive_forward() 
        elif self.state == 'BACKWARD':
            self.drive_backward()

    def turn_lelf(self, degree_to_turn):
        initial_encoder_value = self.get_current_encoder()
        while abs(self.calculate_degree(initial_encoder_value)) < degree_to_turn:
            self.left_motor.backward(self.turning_speed)
            self.right_motor.stop()
        
        if self.state == 'FORWARD':
            self.drive_forward() 
        elif self.state == 'BACKWARD':
            self.drive_backward()

    def slide_right(self, degree_to_slide):
        initial_encoder_value = self.get_current_encoder()
        while abs(self.calculate_degree(initial_encoder_value)) < degree_to_slide:
            self.left_motor.forward(self.driving_speed+self.sliding_speed)
            self.right_motor.forward(self.driving_speed-self.sliding_speed)
        
        if self.state == 'FORWARD':
            self.drive_forward() 
        elif self.state == 'BACKWARD':
            self.drive_backward()
    
    def slide_left(self, degree_to_slide):
        initial_encoder_value = self.get_current_encoder()
        while abs(self.calculate_degree(initial_encoder_value)) < degree_to_slide:
            self.right_motor.forward(self.driving_speed+self.sliding_speed)
            self.left_motor.forward(self.driving_speed-self.sliding_speed)
        
        if self.state == 'FORWARD':
            self.drive_forward() 
        elif self.state == 'BACKWARD':
            self.drive_backward()

    def accelerate(self):
        if self.driving_speed + self.acceleration_rate <= 1:
            self.driving_speed += self.acceleration_rate
        else:
            return 

        if self.state == 'FORWARD':
            self.drive_forward() 
        elif self.state == 'BACKWARD':
            self.drive_backward()
            
    def decelerate(self):
        if self.driving_speed - self.acceleration_rate >= 0:
            self.driving_speed -= self.acceleration_rate
        else:
            return 

        if self.state == 'FORWARD':
            self.drive_forward() 
        elif self.state == 'BACKWARD':
            self.drive_backward()
        
    def calculate_degree(self, initial_encoder_value):
        count_per_degree = 6
        current_encoder = self.get_current_encoder()
        left_different = current_encoder['left'] - initial_encoder_value['left']
        right_different = current_encoder['right'] - initial_encoder_value['right']
        degree_different = (left_different - right_different)/count_per_degree
        # print(degree_different)
        return degree_different

    def get_current_encoder(self):
        encoder_value = json.loads(self.redisdb.get('encoder_value').decode('ascii'))
        # print(encoder_value)
        return encoder_value

    def calculate_distance(self, initial_encoder_value):
        count_per_cm = 20
        current_encoder = self.get_current_encoder()
        left_different = current_encoder['left'] - initial_encoder_value['left']
        right_different = current_encoder['right'] - initial_encoder_value['right']
        distance_different = (left_different + right_different)/ (count_per_cm*2)
        # print(distance_different)
        return abs(distance_different)

    def forward_distance(self, distance_to_run):
        initial_encoder_value = self.get_current_encoder()
        while self.calculate_distance(initial_encoder_value) < distance_to_run:
            self.drive_forward()
            line_result = self.redisdb.get('line_result').decode('ascii')
            line_result = int(line_result)
            print(line_result)
            if line_result < 1:
                print('need to slide right')
                print(line_result)
                self.slide_right(0.5)
            elif line_result > 1:
                print('need to slide left')
                print(line_result)
                self.slide_left(0.5)

        
        self.stop()
    
    def forward_distance_without_line(self, distance_to_run):
        self.drive_forward()
        initial_encoder_value = self.get_current_encoder()
        while self.calculate_distance(initial_encoder_value) < distance_to_run:
            degree = self.calculate_degree(initial_encoder_value)
            if degree > 0: 
                self.slide_left(0.5)
                print('need to slide left')
                print(degree)
            elif degree < -0:
                self.slide_right(0.5)
                print('need to slide right')
                print(degree)
            self.drive_forward()
        
        self.stop()

    def keep_the_line(self):
        line_result = self.redisdb.get('line_result').decode('ascii')
        line_result = int(line_result)
        while line_result !=  111:
            self.drive_forward()
            print(line_result)
            if line_result < 0:
                print('need to slide right')
                print(line_result)
                self.slide_right(0.2)
            elif line_result > 0:
                print('need to slide left')
                print(line_result)
                self.slide_left(0.2)
            line_result = self.redisdb.get('line_result').decode('ascii')
            line_result = int(line_result)

    def drive_manual(self, left_speed, right_speed):
        self.left_motor.forward(left_speed)
        self.right_motor.forward(right_speed)

    def get_line_array(self):
        value = self.redisdb.get('line_value').decode('ascii')
        return value

    def get_left_array(self):
        value = self.redisdb.get('line_left').decode('ascii')
        return value

    def get_ball_state(self):
        value = self.redisdb.get('ball_state').decode('ascii')
        return value

    def get_left_ball_ir(self):
        average = 0
        for i in range(20):
            value = int(self.redisdb.get('ball_ir').decode('ascii'))
            average = average + value
        
        return average/20

    def get_front_ir(self):
        average = 0
        for i in range(20):
            value = int(self.redisdb.get('front_ir').decode('ascii'))
            average = average + value
        
        return average/20

    
            
def set_boudary(value, min_value=0, max_value=1):
    if value > max_value:
        return max_value
    elif value < min_value:
        return min_value
    else:
        return value

def main():
    chasis = Chasis()
    drop.off()
    pinA.off()
    pinB.on()

    # From starting point until reach the circle

    k1 = 0
    k2 = 0.5
    k3 = 0.5
    k4 = 1
    run_speed=0.3
    kp=0.1
    found_circle = False
    while not found_circle:
        
        line_value = chasis.get_line_array()
        result = 0
        try:    
            result += int(line_value[0])*-k4
            result += int(line_value[1])*-k3
            result += int(line_value[2])*-k2
            result += int(line_value[3])*-k1
            result += int(line_value[4])*k1
            result += int(line_value[5])*k2
            result += int(line_value[6])*k3
            result += int(line_value[7])*k4
            left_speed = set_boudary(run_speed-kp*result, min_value=0.0)
            right_speed = set_boudary(run_speed+kp*result, min_value=0.0, max_value=1.0)
            zero_found = 0
            for char in line_value:
                if char == '0':
                    zero_found += 1
            # print(zero_found)
            if zero_found == 0:
                found_circle = True
                chasis.turn_lelf(87)
                chasis.stop()
                time.sleep(1)
                chasis.drive_forward()
                
            
            current_speed = {
                'left': left_speed,
                'right': right_speed
            }
            chasis.redisdb.set('driving_speed', json.dumps(current_speed))
            chasis.drive_manual(left_speed, right_speed)
        except Exception as e:
            print(e)

    # ---------------------------------

    # try to collect the balls while moving around
    intersection_found = 0
    line_status = 'on'
    drop_ball = True
    ball_found = 0
    last_stage = False
    last_intersection_time = time.time()
    k1 = 0
    k2 = 0.5
    k3 = 1
    k4 = 2
    run_speed=0.3
    kp=0.1
    while not last_stage:
        
        line_value = chasis.get_line_array()
        # print(line_value)
        result = 0
        try:    
            result += int(line_value[0])*-k4
            result += int(line_value[1])*-k3
            result += int(line_value[2])*-k2
            result += int(line_value[3])*-k1
            result += int(line_value[4])*k1
            result += int(line_value[5])*k2
            result += int(line_value[6])*k3
            result += int(line_value[7])*k4
            left_speed = set_boudary(run_speed-kp*result, min_value=0.3)
            right_speed = set_boudary(run_speed+kp*result, min_value=0.0, max_value=0.5)

            left_line = chasis.get_left_array()
            if left_line[0] == '1' or left_line[1] == '1':
                if time.time() - last_intersection_time > 2:
                    if time.time() - last_intersection_time < 3 and intersection_found > 0:
                        intersection_found = 0
                    else:
                        intersection_found = intersection_found+1
        
                    print("I found : ", intersection_found, " intersection")
                    if intersection_found == 2 and drop_ball == True:
                        
                        chasis.stop()
                        last_stage = True
                        chasis.turning_speed = 0.2
                        initial_encoder_value = chasis.get_current_encoder()
                        while abs(chasis.calculate_degree(initial_encoder_value)) < 80:
                            chasis.right_motor.forward(chasis.turning_speed)
                            chasis.left_motor.backward(chasis.turning_speed)
                            
                        chasis.stop()
                        time.sleep(1)
                        chasis.driving_speed = 0.2
                        chasis.drive_forward()
                    last_intersection_time = time.time()

            ball_ir = chasis.get_left_ball_ir()
            if ball_ir < 15:
                # print(ball_ir)
                chasis.turn_right(60)
                chasis.stop()
                time.sleep(1)
                chasis.drive_forward()
                time.sleep(2)

                    
            ball_state = chasis.get_ball_state()
            if ball_state == '100':
                print('Found the ball!!')
                time.sleep(1)
                chasis.stop()
                time.sleep(15)
                ball_found = ball_found + 1
                if ball_found > 3:
                    drop_ball = True
                
            if line_value == '11111111':
                line_status = 'off'
                print("lose line")
                left_speed = 0.45
                right_speed = 0.2
            else:
                if line_status == 'off':
                    print('back on-line')
                    line_status = 'on'
                    chasis.stop()
                    time.sleep(1)
                    chasis.drive_forward()
            
            current_speed = {
                'left': left_speed,
                'right': right_speed
            }
            chasis.redisdb.set('driving_speed', json.dumps(current_speed))
            chasis.drive_manual(left_speed, right_speed)
        except Exception as e:
            print(e)


    while chasis.get_front_ir() > 7:
        ''' try to reach Ze-bot and drop the ball'''

        line_value = chasis.get_line_array()
        # print(line_value)
        result = 0
        try:    
            result += int(line_value[0])*-k4
            result += int(line_value[1])*-k3
            result += int(line_value[2])*-k2
            result += int(line_value[3])*-k1
            result += int(line_value[4])*k1
            result += int(line_value[5])*k2
            result += int(line_value[6])*k3
            result += int(line_value[7])*k4
            left_speed = set_boudary(run_speed-kp*result, min_value=0.0)
            right_speed = set_boudary(run_speed+kp*result, min_value=0.0, max_value=1.0)
            zero_found = 0
                
            current_speed = {
                'left': left_speed,
                'right': right_speed
            }
            chasis.redisdb.set('driving_speed', json.dumps(current_speed))
            chasis.drive_manual(left_speed, right_speed)
        except Exception as e:
            print(e)
    
    drop.on()
    pinA.off()
    pinB.on()
    print("!!END!!")

if __name__ == '__main__':
    main()     

