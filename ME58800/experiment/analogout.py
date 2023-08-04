from gpiozero import LED

drop = LED(17)
pinA = LED(27)
pinB = LED(22)

while True:
    drop.on()
    pinA.off()
    pinB.on()