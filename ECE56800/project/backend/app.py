'''
    AUTHOR: Thirawat Bureetes
    EMAIL: tbureete@purdue.edu
    DETAIL: This file will create REST API web server.
'''

from flask import Flask
from flask_restful import Api
from flask_cors import CORS

from sensor import RGBRequest

app = Flask(__name__)
CORS(app)
api = Api(app)

api.add_resource(RGBRequest, '/rgb')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)