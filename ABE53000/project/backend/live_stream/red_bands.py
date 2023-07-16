import numpy as np
import cv2

LEFT_PATH = "/home/matthew/Pictures/stereo/capture/left/{:06d}.jpg"
RIGHT_PATH = "/home/matthew/Pictures/stereo/capture/right/{:06d}.jpg"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
left = cv2.VideoCapture(1)
# set left frames per second:
left.set(cv2.CAP_PROP_FPS, 1)

right = cv2.VideoCapture(2)
# set right frames per second:
right.set(cv2.CAP_PROP_FPS, 1)


# Increase the resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

frameId = 0

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    if not (left.grab() and right.grab()):
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    leftFrame = cropHorizontal(leftFrame)
    _, rightFrame = right.retrieve()
    rightFrame = cropHorizontal(rightFrame)

    cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
    cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)
    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)

    #find the redbands of the left and right cameras for nvdi calculations ///
    redleft = leftFrame[:, :, 2]
    redright = rightFrame[:, :, 2]
    cv2.imshow('left_red', redleft)
    cv2.imshow('right_red', redright)

    #create a live NDVI heat map. ///
    #NDVI = (NIR - RED)/(NIR + RED) ///
    #Use left redband for RED and use right redband for NIR ///
    bottom = (redleft.astype(float) + redright.astype(float))
    bottom[bottom == 0] = 0.01 #so we don't divide by 0 ///
    ndvi_heat = (redleft.astype(float) - redright.astype(float)) / bottom
    ndvi_heat = ndvi_heat.astype(np.uint8)
    cv2.imshow('NDVI', ndvi_heat)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameId += 1

left.release()
right.release()
cv2.destroyAllWindows()
