#include <Wire.h>
#include <Servo.h>
#include <SoftwareSerial.h>
#include <Adafruit_TCS34725.h>

// define pin for ble module
SoftwareSerial BTSerial(6, 7);

// define color sensor
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_700MS, TCS34725_GAIN_1X);

// define state for pickup and sorting ball
const int S0 = 100;
const int S1 = 101;
const int S2 = 102;
const int S3 = 103;
int state = S0;

// define neccessary variable
boolean Ball = false; 
boolean know_ball = false;
int ball_type = 0; // lacrose = 1, pool = 2, tennis = 3
boolean drop_ball = false;
int pos;
int average_coeff = 6;
int counter;

// pin for recieve command from rpi
int pinA = 2;
int pinB = 3;
int dropPin = 4;

//define angle for servos
const int arm_low = 20; //arm start position
const int arm_mid = 85;  //classification position
const int arm_high = 138;  //drop ball positon
const int div_lac = 145; // tested at 145 degrees
const int div_pool = 25; // tested at 30 degrees
const int div_tennis = 90; // tested at 90 degrees
const int drop_start = 180; //tested to stop balls from rolling out
const int drop_tennis = 168; //tested drop at 168 degrees
const int drop_pool = 154; //tested drop at 154 degrees
const int drop_lac = 130; //tested at 130 degrees

// define servos
Servo servo_arm; 
Servo servo_div;
Servo servo_drop;

void setup() {

  pinMode(pinA, INPUT);
  pinMode(pinB, INPUT);
  pinMode(dropPin, INPUT);

  servo_arm.attach(9);
  servo_div.attach(10);
  servo_drop.attach(11);

  servo_arm.write(arm_low);
  servo_drop.write(drop_start);

  if (tcs.begin()) {
    Serial.println("Found sensor");
  } else {
    Serial.println("No TCS34725 found ... check your connections");
    while (1);
  }
  
  Serial.begin(9600);
  Wire.begin();
  BTSerial.begin(9600);
}
 
void loop() {
//  variable for color sensor
  uint16_t r, g, b, c, colorTemp, lux, ball_color, red, green;
  
// get data from color sensor
  tcs.getRawData(&r, &g, &b, &c);

// receive command from rpi
  if(digitalRead(pinA) == LOW && digitalRead(pinB) == LOW){
  }
  else if(digitalRead(pinA) == HIGH && digitalRead(pinB) == LOW){
    ball_type = 1;
  }
  else if(digitalRead(pinA) == LOW && digitalRead(pinB) == HIGH){
    ball_type = 2;
  }
  else if(digitalRead(pinA) == HIGH && digitalRead(pinB) == HIGH){
    ball_type = 3;
  }
  
  if (digitalRead(dropPin)== HIGH){
    drop_ball = true;
    if(ball_type == 1) {
      servo_drop.write(drop_lac);
    }
    else if(ball_type == 2) {
      servo_drop.write(drop_pool);
    }
    else if(ball_type == 3) {
      servo_drop.write(drop_tennis);
    }
  }
  else{
    drop_ball = false;
    servo_drop.write(drop_start);
    if(ball_type == 1) {
      BTSerial.write("ZB+POSL");
    }
    else if(ball_type == 2) {
      BTSerial.write("ZB+POSC");
    }
    else if(ball_type == 3) {
      BTSerial.write("ZB+POST");
    }
  }

//  use c paremeter from color sensor to identify ball detection
  if (c > 680){
    Serial.print("Ball:");
    Serial.println(state);
    Ball = true;
  }
  
  switch(state){
//  base state
    case S0:
      if (Ball){
        state = S1;
      }
      else{
        state = S0;
      }
      break;
    case S1:
//    find the ball and lefting the ball
      if(pos!=arm_mid)
      {
        for (pos = arm_low; pos < arm_mid; pos += 1) {
          servo_arm.write(pos);
          delay(20);
        }
      }
      delay(500);
      ball_color=0;
      lux = 0;
      red = 0;
      green = 0;
//    avarage value from color sensor
      for (counter=0; counter < average_coeff; counter++)
      {
        tcs.getRawData(&r, &g, &b, &c);
        ball_color=ball_color+c;
        lux += tcs.calculateLux(r, g, b);
        red += r;
        green +=g;
        delay(20);
      }

      ball_color=ball_color/average_coeff;
      lux /= average_coeff;
      red /= average_coeff;
      green /= average_coeff;
      
//    classify ball type
      if(ball_color<2000)
      {ball_type=2;}
      else{
        if(float (red)/green < 1.07)
        {ball_type=3;}
        else
        {ball_type=1;}
      }
      state=S2;
      break;
    case S2:
//    adjust position of ball sorting servo
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
//    lift ball to top of robot
      if(pos!=arm_high)
      {
        for (pos = arm_mid; pos < arm_high; pos += 1) { 
          servo_arm.write(pos);
          delay(20);
        }
      }
      delay(1000);
      for (; pos > arm_low; pos -= 1) { 
          servo_arm.write(pos);
          delay(20);
        }
        delay(1000);
        state = S0;
        Ball = false;
      break;
  }
  delay(15);                  
}
