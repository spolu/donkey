import socketio
from aiohttp import web

sio = socketio.Server()

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    global timer
    timer.reset()
    send_control(0, 0)

def main():
    global app

    address = ('0.0.0.0', 9090)
    app = socketio.Middleware(sio, app)

    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print('Interrupted')

if __name__ == "__main__":
    main()
