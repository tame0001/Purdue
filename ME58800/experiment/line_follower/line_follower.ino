#include <Wire.h>
unsigned char t;
//void send_data(short a1,short b1,short c1,short d1,short e1,short f1);
unsigned char data[16];
String result;
int theshore =  70;
void setup()
{
  Wire.begin();        // join i2c bus (address optional for master)
  Serial.begin(112500);  // start serial for output
  t = 0;
  Serial.println("Setup");
}
void loop()
{
  Wire.requestFrom(9, 16);    // request 16 bytes from slave device #9
  Serial.println("Reading Line Sensor");
  while (Wire.available())   // slave may send less than requested
  {
    data[t] = Wire.read(); // receive a byte as character
    if (t < 15)
      t++;
    else
      t = 0;
  }

//  for (t=0; t < 16; t++){
//    if (t%2 == 0){
//      Serial.println(data[t]);
//    }
//////      if(data[t] < theshore){
//////         result[t/2] = '1';
//////      }
//////      else{
//////        result[t/2] = '0';
//////      }
//////    }
//  }
  Serial.print("data[0]:");
  Serial.println(data[0]);
  Serial.print("data[2]:");
  Serial.println(data[2]);
  Serial.print("data[4]:");
  Serial.println(data[4]);
  Serial.print("data[6]:");
  Serial.println(data[6]);
  Serial.print("data[8]:");
  Serial.println(data[8]);
  Serial.print("data[10]:");
  Serial.println(data[10]);
  Serial.print("data[12]:");
  Serial.println(data[12]);
  Serial.print("data[14]:");
  Serial.println(data[14]);

  result = "";
  for(int i=0; i<8; i++){
    if(data[i*2] > theshore){
    Serial.println("1");
    result.concat("1");
    }
    else{
      Serial.println("0");
      result.concat("0");
    }
  }
  Serial.println(result);
  
  delay(1000);
}
