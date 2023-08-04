#include <SharpIR.h>


// These constants won't change. They're used to give names to the pins used:
const int analogInPin = A0;  // Analog input pin that the potentiometer is attached to
int distance;

SharpIR sensor( SharpIR::GP2Y0A41SK0F, analogInPin );

void setup() {
  // initialize serial communications at 9600 bps:
  Serial.begin(115200);
  
}

void loop() {
  // read the analog in value:
//  sensorValue = analogRead(analogInPin);
  // map it to the range of the analog out:
//  outputValue = map(sensorValue, 0, 1023, 0, 255);
  // change the analog out value:
//  analogWrite(analogOutPin, outputValue);

  distance = sensor.getDistance(); //Calculate the distance in centimeters and store the value in a variable

  // print the results to the Serial Monitor:
//  Serial.print("sensor = ");
//  Serial.print(sensorValue);
//  Serial.print("\t output = ");
//  Serial.println(outputValue);
  Serial.println(distance);

  // wait 2 milliseconds before the next loop for the analog-to-digital
  // converter to settle after the last reading:
  delay(2);
}
