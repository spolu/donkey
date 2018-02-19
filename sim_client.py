import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

sio = socketio.Server()
app = Flask(__name__)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

@sio.on('telemetry')
def telemetry(sid, data):
    print("telemetry", sid)
    print("DATA: {}".format(data))

    sio.emit('step', data={}, skip_sid=True)

def main():
    address = ('0.0.0.0', 9090)
    global app
    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print('stopping')

if __name__ == "__main__":
    main()
