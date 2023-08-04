/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 http://www.arduino.cc/en/Tutorial/Sweep
*/

#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 19;    // variable to store the servo position

void setup() {
  myservo.attach(11);  // attaches the servo on pin 9 to the servo object
//  Serial.begin(9600);
//  for (pos = 132; pos > 19; pos -= 1) { // goes from 180 degrees to 0 degrees
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
//    delay(15);                       // waits 15ms for the servo to reach the position
//  }
  myservo.write(180); 
  delay(1000);
//  for (pos = 19; pos < 80; pos += 1) { // goes from 0 degrees to 180 degrees
//    // in steps of 1 degree
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
//    delay(20);                       // waits 15ms for the servo to reach the position
//  }
//  delay(1000);
// for (; pos < 132; pos += 1) { // goes from 0 degrees to 180 degrees
//    // in steps of 1 degree
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
//    delay(20);                       // waits 15ms for the servo to reach the position
//  }
//  delay(2000);
//  for (pos = 132; pos > 20; pos -= 1) { // goes from 180 degrees to 0 degrees
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
//    delay(15);                       // waits 15ms for the servo to reach the position
//  }
}

void loop() {
//  Serial.println(Serial.read());
//  myservo.write(19);
  
//  for (pos; pos > 19; pos -= 1) { // goes from 180 degrees to 0 degrees
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
//    delay(15);                       // waits 15ms for the servo to reach the position
//  }
}
