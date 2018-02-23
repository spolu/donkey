import threading
import time
import tty, termios, sys
import simulation

cmd = simulation.Command(0.0, 0.0, 0.0)
reset = False

def getchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def input_thread():
    global cmd
    global reset

    while True:
        key = getchar()
        if key.strip() == 'q':
            break
        if key == 'w':
            cmd = simulation.Command(0.0, 1.0, 0.0)
        if key == 's':
            cmd = simulation.Command(0.0, 0.0, 0.0)
        if key == 'a':
            cmd = simulation.Command(-0.8, 0.0, 0.0)
        if key == 'd':
            cmd = simulation.Command(0.8, 0.0, 0.0)
        if key == 'b':
            cmd = simulation.Command(0.0, 0.0, 1.0)
        if key == 'r':
            reset = True;

        print("key", key)

def main():
    global cmd
    global reset

    threading.Thread(target = input_thread).start()

    c = simulation.Simulation(launch=True, headless=False)
    c.start()

    while True:
        if reset:
            reset = False
            c.reset()
        else:
            c.step(cmd)
            print(c.telemetry())

if __name__ == "__main__":
    main()
