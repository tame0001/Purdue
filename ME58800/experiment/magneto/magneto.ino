#include <Wire.h>
#include <LIS3MDL.h>
#include <math.h>

LIS3MDL mag;

double heading;
char report[100];

void setup()
{
  Serial.begin(112500);
  Wire.begin();

  if (!mag.init())
  {
    Serial.println("Failed to detect and initialize magnetometer!");
    while (1);
  }

  mag.enableDefault();
}

void loop()
{
  mag.read();
  heading = 180 * atan2(double(mag.m.y), double(mag.m.x)) / M_PI;
  snprintf(report, sizeof(report), "M: %6d %6d %6d",
    mag.m.x, mag.m.y, mag.m.z);
  Serial.println(report);
  Serial.println(heading);

  delay(100);
}
