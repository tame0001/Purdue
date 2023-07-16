from flask import Flask
from flask_restful import Api
from flask_cors import CORS

from capture import CaptureRequest, LiveRequest, CalibrateRequest

app = Flask(__name__)
CORS(app)
api = Api(app)

api.add_resource(CaptureRequest, '/capture')
api.add_resource(LiveRequest, '/live')
api.add_resource(CalibrateRequest, '/calibrate')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)