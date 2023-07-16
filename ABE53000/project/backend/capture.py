from flask_restful import Resource, reqparse
from PIL import Image
import picamera
import picamera.array
import subprocess
import os
import time
import cv2 
import redis
import json
import xml.etree.ElementTree as xml

class CaptureRequest(Resource):
    def post(self):
        return {'code': 0}, 201

    def get(self):
        try:
            folder = '/home/pi/images/'
            file_format = '.png'
            redisdb = redis.Redis(host='localhost', port=6379, db=0)
            last_filename = redisdb.get('filename')
            if last_filename is None:
                # print('find the new file name')
                stamp = 1
                while os.path.exists(folder + str(stamp) + file_format):
                    stamp = stamp+1
            # os.makedirs(folder + str(stamp))
            else:
                stamp = int(redisdb.get('stamp')) + 1   
            filename = folder + str(stamp) + file_format
            redisdb.set('stamp', stamp)
            redisdb.set('filename', filename)
            redisdb.set('capture_request', 'True')
            # print(filename)
            
        except Exception as error:
            print(error)
            print(type(error))
            return {'code': 1}, 200
        else:
            while redisdb.get('capture_request') is not None:
                time.sleep(0.1)
            return {
                'code': 0,
                'filename': 'images/' + str(stamp) + file_format
            }, 200

class LiveRequest(Resource):
    def get(self):
        redisdb = redis.Redis(host='localhost', port=6379, db=0)
        if redisdb.get('live_request') is not None:
            redisdb.delete('live_request')
        else:
            redisdb.set('live_request', 'True')
        return {
                'code': 0,
            }, 200 

class CalibrateRequest(Resource):
    def get(self):
        redisdb = redis.Redis(host='localhost', port=6379, db=0)
        config = json.loads(redisdb.get('config').decode('ascii'))
        if redisdb.get('calibrating') is not None:
            redisdb.delete('calibrating')
            config_file = xml.parse('live_stream/config.xml')
            cameras = config_file.getroot().findall('camera') 
            for camera in cameras:
                if camera.attrib['type'] == 'nir':
                    for border in ('top', 'bottom', 'right', 'left'):
                        value = camera.find(border)
                        value.text = str(config['nir'][border])
            config_file.write('live_stream/config.xml')
        else:
            redisdb.set('calibrating', 'True')
        return {
                'code': 0,
            }, 200

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('code')
        args = parser.parse_args()
        # print(args['code'])
        redisdb = redis.Redis(host='localhost', port=6379, db=0)
        # print(redisdb.get('config'))
        # print(json.loads(redisdb.get('config').decode('ascii')))
        config = json.loads(redisdb.get('config').decode('ascii'))
        resulotion = (320, 240)
        leverage = 5
        # print(type(args['code']))
        if args['code'][0] == '2':
            change = (leverage-1) * int(args['code'][2]) + 1
            if args['code'][1] == '1' and config['nir']['bottom'] + change <= resulotion[1]:
                config['nir']['top'] = config['nir']['top'] + change
                config['nir']['bottom'] = config['nir']['bottom'] + change
            elif args['code'][1] == '3' and config['nir']['top'] - change >=0:
                config['nir']['top'] = config['nir']['top'] - change
                config['nir']['bottom'] = config['nir']['bottom'] - change
            elif args['code'][1] == '4' and config['nir']['left'] - change >=0:
                config['nir']['left'] = config['nir']['left'] - change
                config['nir']['right'] = config['nir']['right'] - change
            elif args['code'][1] == '2' and config['nir']['right'] + change <= resulotion[0]:
                config['nir']['left'] = config['nir']['left'] + change
                config['nir']['right'] = config['nir']['right'] + change
        print(config)
        redisdb.set('config', json.dumps(config))
        return {'code': 0}, 201