/*
  Motor - PID speed control
  (1) Receive command from Visual Studio (via COM4): set_speed, kP, kI, kD
  (2) Control motor speed through PWM (PWM is base on PID calculation)
  (3) Send pv_speed to Visual Studio -> show in graph
  
 Created 31 Dec. 2016
 This example code is in the public domain.

 http://engineer2you.blogspot.com
 */
String mySt = "";
char myChar;
boolean stringComplete = false;  // whether the string is complete


// pin interface
const byte pin_a = 2;   //for encoder pulse A
const byte pin_b = 3;   //for encoder pulse B
const byte pin_fwd = 4; //for H-bridge: run motor forward
const byte pin_bwd = 5; //for H-bridge: run motor backward
const byte pin_pwm = 6; //for H-bridge: motor speed
//

boolean motor_start = false;
boolean positionMode = false;
double encoder = 0;
double counterPerLoop = 0;
int m_direction = 0;
int sv_speed = 100;     //this value is 0~255
double pv_speed = 0;
double set_speed = 0;
double set_position = 0;
double start_position = 0;
double distance = 0;
double e_speed = 0; //error of speed = set_speed - pv_speed
double e_speed_pre = 0;  //last error of speed
double e_speed_sum = 0;  //sum error of speed
double pwm_pulse = 0;     //this value is 0~255
double kp = 2.3;
double ki = 0.08;
double kd = 0.3;
int timer1_counter; //for timer
int i=0;


void setup() {
  pinMode(pin_a,INPUT_PULLUP);
  pinMode(pin_b,INPUT_PULLUP);
  pinMode(pin_fwd,OUTPUT);
  pinMode(pin_bwd,OUTPUT);
  pinMode(pin_pwm,OUTPUT);
  attachInterrupt(digitalPinToInterrupt(pin_a), detect_a, RISING);
  // start serial port at 9600 bps:
  Serial.begin(112500);
  //--------------------------timer setup
  noInterrupts();           // disable all interrupts
  TCCR1A = 0;
  TCCR1B = 0;
  timer1_counter = 59286;   // preload timer 65536-16MHz/256/2Hz (34286 for 0.5sec) (59286 for 0.1sec)

  
  TCNT1 = timer1_counter;   // preload timer
  TCCR1B |= (1 << CS12);    // 256 prescaler 
  TIMSK1 |= (1 << TOIE1);   // enable timer overflow interrupt
  interrupts();             // enable all interrupts
  //--------------------------timer setup
  
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  
  analogWrite(pin_pwm,0);   //stop motor
  digitalWrite(pin_fwd,0);  //stop motor
  digitalWrite(pin_bwd,0);  //stop motor
  
  double kp = 2.3; //debug kp
  double ki = 0.2; //debug ki
  double kd = 0.5; //debug kd
}

void loop() {
  if (stringComplete) {
    // clear the string when COM receiving is completed
    mySt = "";  //note: in code below, mySt will not become blank, mySt is blank until '\n' is received
    stringComplete = false;
  }

  motor(mySt);
}

/*input: cmd
fw: start forvward moving
rv: start reverse moving
st: stop moving
psnum: set moving distance num
spnum: set moving speed num
kpnum: set kp with num while tuning the parameters
kinum: set ki with num while tuning the parameters
kdnum: set kd with num while tuning the parameters 

there are two moving mode, one is position moving mode,
the other one is speed moving mode.

position moving mode:
motor('psnum') to set moving position num, num is unsigned num, example ps100
motor('spnum') to set moving speed num, example sp400
motor('fw') or motor('rv') to set moving direction and start moving
the motor will stop when arriving the position

speed moving mode:
motor('spnum') to set moving speed num
motor('fw') or motor('rv') to set moving direction and start moving
motor('st') to stop moving

motor('st') can stop the motor at any time and will clear sp and ps paramters after that

*/
void motor(String cmd){
  
  if (cmd.substring(0,2) == "fw"){
    digitalWrite(pin_fwd,1);      //run motor run forward
    digitalWrite(pin_bwd,0);
    motor_start = true;
  }
  if (cmd.substring(0,2) == "rv"){
    digitalWrite(pin_fwd,0);      //run motor run reverse
    digitalWrite(pin_bwd,1);
    motor_start = true;
  }
  if (cmd.substring(0,2) == "st"){
    digitalWrite(pin_fwd,0);
    digitalWrite(pin_bwd,0);      //stop motor
    set_speed = 0;
    set_position = 0;
    motor_start = false;
  }
  if (cmd.substring(0,2) == "ps"){
    set_position = cmd.substring(2,cmd.length()).toFloat();  //get set position
    positionMode = true;
    start_position = encoder;
  }
  if (cmd.substring(0,2) == "sp"){
    set_speed = cmd.substring(2,cmd.length()).toFloat();  //get string after set_speed
  }
  if (cmd.substring(0,2) == "kp"){
    kp = cmd.substring(2,cmd.length()).toFloat(); //get string after vs_kp
  }
  if (cmd.substring(0,2) == "ki"){
    ki = cmd.substring(2,cmd.length()).toFloat(); //get string after vs_ki
  }
  if (cmd.substring(0,2) == "kd"){
    kd = cmd.substring(2,cmd.length()).toFloat(); //get string after vs_kd
  }  
}
// encoder counting pulse
void detect_a() {
  encoder+=1; //increasing encoder
  counterPerLoop+=1; //increasing counter at new pulse
  m_direction = digitalRead(pin_b); //read direction of motor
}

// motor controlling algrithm
ISR(TIMER1_OVF_vect)        // interrupt service routine - tick every 0.1sec
{
  TCNT1 = timer1_counter;   // set timer
  pv_speed = (counterPerLoop)*10;  //calculate motor speed, unit is rpm
  counterPerLoop=0;
  //motor_start = 1;
  //set_speed = 10;
  //print out speed
  if (Serial.available() <= 0) {
    Serial.print("status:");
    Serial.print(motor_start);
    Serial.print(" setspeed:");
    Serial.print(set_speed);
    Serial.print(" reaLspeed:");
    Serial.print(pv_speed);
    Serial.print(" dist:");
    Serial.print(distance);
    Serial.print(" kp:");
    Serial.print(kp);
    Serial.print(" ki:");
    Serial.print(ki);
    Serial.print(" kd:");
    Serial.println(kd);
    }
  //PID program
  if (motor_start){
    if(positionMode){
      distance = abs(start_position-encoder);
      if(distance > set_position){
        motor_start = 0;
        set_position = 0;
        set_speed = 0;
      }
    }
    e_speed = set_speed - pv_speed;
    pwm_pulse = double(e_speed)*kp/3.0 + double(e_speed_sum)*ki/3.0 + double(e_speed - e_speed_pre)*kd/3.0;
    e_speed_pre = e_speed;  //save last (previous) error
    e_speed_sum += e_speed; //sum of error
    if (e_speed_sum >12000) e_speed_sum = 12000;
    if (e_speed_sum <-40000) e_speed_sum = -40000;
  }
  else{
    e_speed = 0;
    e_speed_pre = 0;
    e_speed_sum = 0;
    pwm_pulse = 0;
  }
  

  //update new speed
  if (pwm_pulse <255 & pwm_pulse >0){
    analogWrite(pin_pwm,pwm_pulse);  //set motor speed  
  }
  else{
    if (pwm_pulse>255){
      analogWrite(pin_pwm,255);
    }
    else{
      analogWrite(pin_pwm,0);
    }
  }
  
}
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    if (inChar != '\n') {
      mySt += inChar;
    }
    // if the incoming character is a newline, set a flag
    // so the main loop can do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
