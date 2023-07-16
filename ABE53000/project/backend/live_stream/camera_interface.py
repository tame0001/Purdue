from imutils.video import VideoStream
from PIL import Image
import imutils
import cv2
import time
import numpy
import sys
import redis
import re
import json
import xml.etree.ElementTree as xml

class Camera(object):
    def __init__(self):
        self.width = 320
        self.height = 240
        self.resolution = (self.width, self.height)
        self.framerate = 25
        self.picam = VideoStream(
            usePiCamera=True, 
            resolution=self.resolution, 
            framerate=self.framerate
            )
        self.webcam = VideoStream(
            usePiCamera=False, 
            framerate=self.framerate
            )
        self.picam.start()
        self.webcam.start()
        time.sleep(2)
        self.redisdb = redis.Redis(host='localhost', port=6379, db=0)
        self.redisdb.delete('filename')
        self.redisdb.delete('capture_request')
        self.redisdb.delete('live_request')
        self.redisdb.delete('calibrating')
        config_file = xml.parse('config.xml').getroot()
        self.config = {}
        for camera in config_file:
            subconfig = {}
            for setting in camera:
                try:
                    subconfig[setting.tag] = int(setting.text)
                except ValueError:
                    subconfig[setting.tag] = float(setting.text)
            self.config[camera.attrib['type']] = subconfig
        # print(self.config)
        # print(json.dumps(self.config))
        self.redisdb.set('config', json.dumps(self.config))
    
    def get_frame(self):
        picam_frame_raw = self.picam.read()
        webcam_frame = self.webcam.read()
        webcam_frame = imutils.resize(webcam_frame, width=self.width)
        picam_frame_raw = imutils.rotate_bound(picam_frame_raw, 270)
        webcam_frame = imutils.rotate_bound(webcam_frame, 270)
        webcam_frame_red = webcam_frame[:,:,2] # extract red band from rgb image
        picam_frame_gray = cv2.cvtColor(picam_frame_raw, cv2.COLOR_BGR2GRAY)
        picam_frame_gray = cv2.normalize(picam_frame_gray, None, alpha=0, beta=255, 
            norm_type=cv2.NORM_MINMAX)
        webcam_frame_red = cv2.normalize(webcam_frame_red, None, alpha=0, beta=255, 
            norm_type=cv2.NORM_MINMAX)
        # print(picam_frame.shape, webcam_frame.shape)
        webcam_frame = cv2.resize(webcam_frame, (0,0), 
            fx=self.config['rgb']['scale'], fy=self.config['rgb']['scale'])
        webcam_frame_red = cv2.resize(webcam_frame_red, (0,0), 
            fx=self.config['rgb']['scale'], fy=self.config['rgb']['scale'])
        picam_frame = picam_frame_raw[self.config['nir']['left']:self.config['nir']['right'], 
            self.config['nir']['top']:self.config['nir']['bottom']]
        picam_frame_gray = picam_frame_gray[self.config['nir']['left']:self.config['nir']['right'], 
            self.config['nir']['top']:self.config['nir']['bottom']]
        # print(picam_frame.shape, webcam_frame.shape)
        combine_frame = numpy.concatenate((webcam_frame, picam_frame), axis=1)
        if self.redisdb.get('capture_request') is not None:
            filename = self.redisdb.get('filename').decode('ascii')
            b,g,r = cv2.split(combine_frame)
            rgb_img = cv2.merge([r,g,b])
            rgb_img = imutils.rotate_bound(rgb_img, 270)
            save_image = Image.fromarray(rgb_img)
            # print(filename)
            save_image.save(filename)
            re_out = re.findall("(.+)(.\w{3})", filename)[0]
            path = re_out[0]
            file_format = re_out[1]
            height = webcam_frame.shape[1]
            upper_image = Image.fromarray(rgb_img[:height,:,:])
            upper_image.save(path+'_ir'+file_format)
            lower_image = Image.fromarray(rgb_img[height:,:,:])
            lower_image.save(path+'_rgb'+file_format)  
            self.ndvi(webcam_frame_red, picam_frame_gray)   
            ndvi_image =  Image.fromarray(self.ndvi_heat)
            ndvi_image.save(path+'_nvdi'+file_format)  
            fp = open('/home/pi/images/nvdi.txt', 'a')
            fp.writelines(path+'_nvdi'+file_format + ': ' + str(round(self.avg_ndvi, 3)) + 
                ' timestamp:' + time.strftime("%D:%H:%M:%S") + '\r\n')
            fp.close()     
            self.redisdb.delete('capture_request')
        if self.redisdb.get('live_request') is not None:
            if self.redisdb.get('calibrating') is not None:
                self.config = json.loads(self.redisdb.get('config').decode('ascii'))
                self.calibrate_frame = numpy.concatenate((webcam_frame, picam_frame), axis=1)
                live_stream = imutils.resize(self.calibrate_frame, width=330)
            else:
                self.ndvi(webcam_frame_red, picam_frame_gray)
                live_stream = imutils.resize(self.ndvi_heat, width=180)
        else:
            live_stream = imutils.resize(combine_frame, width=310)
        jpeg = cv2.imencode('.jpg', live_stream)[1]
        return jpeg.tobytes()

    def ndvi(self, rgb, nir):
        #NDVI = (NIR - RED)/(NIR + RED) ///
        # rgb_red = cv2.split(rgb)[2]
        # nir_red = cv2.split(nir)[2]
        denom = (rgb.astype(float) + nir.astype(float))
        denom[denom == 0] = 0.01 #so we don't divide by 0 ///
        self.ndvi_heat = (rgb.astype(float) - nir.astype(float)) / denom
        self.avg_ndvi =self.ndvi_heat.mean()
        self.ndvi_heat = (self.ndvi_heat*(2**7)) + 2**7 # span [-1,1] to [0,255]  
        self.ndvi_heat = self.ndvi_heat.astype(numpy.uint8)
        cv2.rectangle(self.ndvi_heat, (110, 210), (160, 240), (255, 255, 255), 30)
        cv2.putText(self.ndvi_heat, str(round(self.avg_ndvi, 3)), (100, 220), 
            cv2.FONT_ITALIC, 0.75, (0, 0, 255), 4)
        # print(self.ndvi_heat)