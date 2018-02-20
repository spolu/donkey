import threading
import time
import socketio
import eventlet
import eventlet.wsgi
import tty, termios, sys

from flask import Flask

sio = socketio.Server()
app = Flask(__name__)

step = {
    "steering": "0.0",
    "throttle": "0.0",
    "break": "0.0",
}

def getchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    sio.emit('reset', data={}, skip_sid=True)

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

@sio.on('telemetry')
def telemetry(sid, data):
    print("telemetry", sid)
    print("position DATA: {}", format(data['position']))
    # with open("donkey_path.txt", "a") as myfile:
    #     position = format(data['position']['x']) + "," + format(data['position']['y']) + "," + format(data['position']['z']) + "," + format(data['time']) + "\n"
    #     myfile.write(position)

    global step
    print ("SENDING: {}".format(step))
    sio.emit('step', step, skip_sid=True)

def input_thread():
    global step
    while True:
        key = getchar()
        if key.strip() == 'q':
            break
        if key == 'w':
            step = {
                "steering": "0.0",
                "throttle": "1.0",
                "break": "0.0",
            }
        if key == 's':
            step = {
                "steering": "0.0",
                "throttle": "0.0",
                "break": "1.0",
            }
        if key == 'a':
            step = {
                "steering": "-5.0",
                "throttle": "0.0",
                "break": "0.0",
            }
        if key == 'd':
            step = {
                "steering": "5.0",
                "throttle": "0.0",
                "break": "0.0",
            }
        print("key", key)

def main():
    address = ('0.0.0.0', 9090)
    global app
    app = socketio.Middleware(sio, app)
    threading.Thread(target = input_thread).start()
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print('stopping')

if __name__ == "__main__":
    main()
