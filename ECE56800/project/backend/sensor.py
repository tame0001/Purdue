'''
    AUTHOR: Thirawat Bureetes
    EMAIL: tbureete@purdue.edu
    DETAIL: This file will access to Redis database and return JSON value for REST API.
'''

from flask_restful import Resource, reqparse
import time
import redis
import json

class RGBRequest(Resource):
    def post(self):
        return {'code': 0}, 201

    def get(self):
        try:
            redisdb = redis.Redis(host='localhost', port=6379, db=0)
            sensor_value = redisdb.get('sensor')
            red = int(redisdb.get('red'))
            blue = int(redisdb.get('blue'))
            green = int(redisdb.get('green'))
            clear = int(redisdb.get('clear'))
            print(sensor_value)
            
        except Exception as error:
            print(error)
            # print(type(error))
            return {'code': 1}, 201
        
        payload = {
                # 'sensor': sensor_value,
                'red': red,
                'green': green,
                'blue': blue,
                'clear': clear
            }
        
        return json.dumps(payload), 200
