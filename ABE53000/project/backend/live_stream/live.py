from flask import Flask, render_template, Response
from flask_cors import CORS

from camera_interface import Camera

app = Flask(__name__)
CORS(app)

@app.route('/hello')
def hello():
    return render_template('hello.html')

@app.route('/live')
def live():
    return render_template('live.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
           mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)