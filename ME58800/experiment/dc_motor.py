from gpiozero import Robot

robot = Robot(left=('GPIO13', 'GPIO20'), right=('GPIO19', 'GPIO16'))
# robot = Motor('GPIO26', 'GPIO20')
# motorB = Motor('GPIO19', 'GPIO16')

speed = 0.8

while(1):
    print("F = Forward")
    print("B = Backward")
    print("L = Left")
    print("R = Right")
    print("A = Accelerate")
    print("D = Decelerate")
    print("S = Stop")
    user_input = input('Enter command: ')
    user_input = user_input.upper()
    if user_input in ['F', 'B', 'L', 'R', 'A', 'D', 'S']:
        if user_input == 'F':
            robot.forward(speed)
        elif user_input == 'B':
            robot.backward(speed)
        elif user_input == 'R':
            robot.right(speed)
        elif user_input == 'L':
            robot.left(speed)
        elif user_input == 'A':
            if speed + 0.1 <= 1:
                speed = speed + 0.1
        elif user_input == 'D':
            if speed + 0.1 >= 0:
                speed = speed - 0.1
        elif user_input == 'S':
            robot.stop()
    else:
        print("Invalid command")