/*
 Controlling a servo position using a potentiometer (variable resistor)
 by Michal Rinott <http://people.interaction-ivrea.it/m.rinott>

 modified on 8 Nov 2013
 by Scott Fitzgerald
 http://www.arduino.cc/en/Tutorial/Knob
*/

#include <Servo.h>
const int S0 = 000;
const int S1 = 001;
const int S2 = 010;
const int S3 = 011;
const int S4 = 100;
int state = S0;
boolean Ball = 0; 
boolean know_ball = 0;
int ball_type = 0; // lacrose = 1, pool = 2, tennis = 3
boolean drop_ball = 0;
const int arm_low = 19; //arm start position
const int arm_mid = 80;  //classification position
const int arm_high = 132;  //drop ball positon
const int div_lac = 145; // tested at 145 degrees
const int div_pool = 30; // tested at 30 degrees
const int div_tennis = 90; // tested at 90 degrees
const int drop_start = 180; //tested to stop balls from rolling out
const int drop_tennis = 168; //tested drop at 168 degrees
const int drop_pool = 154; //tested drop at 154 degrees
const int drop_lac = 130; //tested at 130 degrees
int count = 0; // number of balls picked up
int drop = 0;

Servo servo_arm;  // create servo object to control a servo
Servo servo_div;
Servo servo_drop;

int potpin = 0;  // analog pin used to connect the potentiometer
int val = 0;    // variable to read the value from the analog pin

void setup() {
  servo_arm.attach(9);  // attaches the servo arm to pin 9 to the servo object
  servo_div.attach(10);
  servo_drop.attach(11);
  //Serial.begin(9600);
}

void loop() {
  switch(state){
    case S0:
      if (Ball){
        state = S1;
      }
      else{
        state = S0;
      }
      break;
    case S1:
      servo_arm.write(arm_mid);
      delay(20); // determine ball from sensor
      // increment ball_type to proper value, 1,2,or 3
      if (know_ball == 1){
        state = S2;
      }
      else{
        state = S1;
      }
      break;
    case S2:
      if(ball_type == 1){
        servo_div.write(div_lac);
        state = S3;
      }
      else if(ball_type == 2){
        servo_div.write(div_pool);
        state = S3;
      }
      else if(ball_type == 3){
        servo_div.write(div_tennis);
        state = S3;
      }
      else{
        state = S2;
      }
      break;
    case S3:
      if(count < 5){
        servo_arm.write(arm_high); //brings arm to the top, dropping the ball
        delay(100); // alloy time for ball to drop
        servo_arm.write(arm_low); //returns the arm to the pickup position
        count = count +1;
      }
      else{
        state = S4;
      }
      break;
    case S4:
    if (drop == 1){
      servo_drop.write(drop_tennis);

      
      // increment drop after meeting Zbot

      
    }
    else if(drop == 2){
      servo_drop.write(drop_pool);

      
      // increment drop after meeting Zbot

      
    }
    else if(drop == 3){
      servo_drop.write(drop_lac);

      
      // increment drop after meeting Zbot

      
    }
    
  }
    
    
  //val=Serial.read();
  //val = analogRead(potpin);            // reads the value of the potentiometer (value between 0 and 1023)
  //val = map(val, 0, 1023, 0, 180);     // scale it to use it with the servo (value between 0 and 180)
  while(val<=180){
    val=val+1;
    
  }
  servo_arm.write(val);                  // sets the servo position according to the scaled value
  //servo.read(9);
  delay(15);                           // waits for the servo to get there
}
