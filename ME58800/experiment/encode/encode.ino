// pins for the encoder inputs
#define L_ENCODER_A 2 
#define L_ENCODER_B 4
#define R_ENCODER_A 3
#define R_ENCODER_B 5
 
// variables to store the number of encoder pulses
// for each motor
volatile long leftCount = 0;
volatile long rightCount = 0;
 
void setup() {
  pinMode(L_ENCODER_A, INPUT);
  pinMode(L_ENCODER_B, INPUT);
  pinMode(R_ENCODER_A, INPUT);
  pinMode(R_ENCODER_B, INPUT);
  
  // initialize hardware interrupts
  attachInterrupt(digitalPinToInterrupt(L_ENCODER_A), leftEncoderEvent, RISING);
  attachInterrupt(digitalPinToInterrupt(R_ENCODER_A), rightEncoderEvent, RISING);
  
  Serial.begin(115200);
}
 
void loop() {
  Serial.print("Right Count: ");
  Serial.println(rightCount);
  Serial.print("Left Count: ");
  Serial.println(leftCount);
  Serial.println();
  delay(3000);
}
 
// encoder event for the interrupt call
void leftEncoderEvent() {
//  if (digitalRead(LH_ENCODER_A) == HIGH) {
//    if (digitalRead(LH_ENCODER_B) == LOW) {
//      leftCount++;
//    } else {
//      leftCount--;
//    }
//  } else {
//    if (digitalRead(LH_ENCODER_B) == LOW) {
//      leftCount--;
//    } else {
//      leftCount++;
//    }
//  }
  
  if (digitalRead(L_ENCODER_B) == LOW){
    leftCount--;
  }
  else{
    leftCount++;
  }
  Serial.println(leftCount);
}
 
// encoder event for the interrupt call
void rightEncoderEvent() {
  if (digitalRead(R_ENCODER_A) == HIGH) {
    if (digitalRead(R_ENCODER_B) == LOW) {
      rightCount++;
    } else {
      rightCount--;
    }
  } else {
    if (digitalRead(R_ENCODER_B) == LOW) {
      rightCount--;
    } else {
      rightCount++;
    }
  }
}
