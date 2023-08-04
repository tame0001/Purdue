#include <Wire.h>
#include <SharpIR.h>
#include <QTRSensors.h>

// pins for the encoder inputs
#define L_ENCODER_A 2 
#define L_ENCODER_B 4
#define R_ENCODER_A 3
#define R_ENCODER_B 5

// define IR distance sensor
const int FrontIRPin = A0;  
const int BallIRPin = A1;
int front_ir_distance;
int ball_ir_distance;
SharpIR front_ir( SharpIR::GP2Y0A41SK0F, FrontIRPin);
SharpIR ball_ir( SharpIR::GP2Y0A21YK0F, BallIRPin);


// define left IR Array
QTRSensors qtr;
const uint8_t LeftIRCount = 2;
uint16_t left_ir_value[LeftIRCount];

// variable to count encoder pusle
volatile long leftCount = 0;
volatile long rightCount = 0;

// variable for right IR Array
unsigned char data[16];
String result;

// theshore for IR Array
const int theshore_ir_array =  70;
const int theshore_left_array =  2000;
int t = 0;
 
void setup() {
  
  pinMode(L_ENCODER_A, INPUT);
  pinMode(L_ENCODER_B, INPUT);
  pinMode(R_ENCODER_A, INPUT);
  pinMode(R_ENCODER_B, INPUT);
  
// initialize hardware interrupts
  attachInterrupt(digitalPinToInterrupt(L_ENCODER_A), leftEncoderEvent, RISING);
  attachInterrupt(digitalPinToInterrupt(R_ENCODER_A), rightEncoderEvent, RISING);

// timer 100 Hz
  TCCR2A = 0;
  TCCR2B = 0;
  TCNT2  = 0;
  OCR2A = 155;
  TCCR2A |= (1 << WGM21);
  TCCR2B |= (1 << CS22) | (1 << CS21) | (1 << CS20);   
  TIMSK2 |= (1 << OCIE2A);

//  initialize left IR Array
  qtr.setTypeRC();
  qtr.setSensorPins((const uint8_t[]){13, 12}, LeftIRCount);
  
  Serial.begin(115200);
  Wire.begin();
}
 
void loop() {

//  read line IR Array on the right side
  Wire.requestFrom(9, 16);    
  while (Wire.available()) 
  {
    data[t] = Wire.read();
    if (t < 15)
      t++;
    else
      t = 0;
  }
  result = "";
  for(int i=0; i<8; i++){
    if(data[i*2] > theshore_ir_array){
    result.concat("1");
    }
    else{
      result.concat("0");
    }
  }
  Serial.print("LineArray:");
  Serial.println(result);

// read IR Reflex sensor on the left side

  qtr.read(left_ir_value);
  result = "";
  Serial.print("LeftIRArray:");
  for (uint8_t i = 0; i < LeftIRCount; i++)
  {
    if(left_ir_value[i] > theshore_left_array){
    result.concat("1");
    }
    else{
      result.concat("0");
    }
  }
  Serial.println(result);
  
  delay(10);
}

ISR(TIMER2_COMPA_vect){
//  DC motor encoders   
  Serial.print("Encoder:");
  Serial.print(leftCount);
  Serial.print(":");
  Serial.println(rightCount);

//  ball IR Sensor 10-80 cm
  ball_ir_distance = ball_ir.getDistance();
  Serial.print("BallIR:");
  Serial.println(ball_ir_distance);
  
//  front IR Sensor 4-30 cm
  front_ir_distance = front_ir.getDistance();
  Serial.print("FrontIR:");
  Serial.println(front_ir_distance);
  
}
 
// encoder event for the interrupt call
void leftEncoderEvent() {  
  if (digitalRead(L_ENCODER_B) == LOW){
    leftCount--;
  }
  else{
    leftCount++;
  }
}
 
// encoder event for the interrupt call
void rightEncoderEvent() {
  if (digitalRead(R_ENCODER_B) == LOW){
    rightCount++;
  }
  else{
    rightCount--;
  }
}
