var socket = io.connect("ws://127.0.0.1:9091")

var keypressed = {
  w: false,
  a: false,
  d: false,
  s: false,
}

var command = () => {
  c = {
    steering: 0.0,
    throttle: 0.0,
    brake: 0.0,
  }

  if (keypressed['w']) {
    c['throttle'] = 1.0
  }
  if (keypressed['a']) {
    c['steering'] = -1.0
  }
  if (keypressed['d']) {
    c['steering'] = 1.0
  }
  if (keypressed['s']) {
    c['brake'] = 1.0
  }

  return c
}

document.addEventListener('keydown', (evt) => {
  keypressed[evt.key] = true
  if (evt.key == 'r') {
    socket.emit('reset', {})
  }
});
document.addEventListener('keyup', (evt) => {
  keypressed[evt.key] = false
});

socket.on('transition', (message) => {
  document.getElementById('reward').innerText = message['reward']
  document.getElementById('progress').innerText = message['progress']
  document.getElementById('time').innerText = message['time']
  document.getElementById('linear_speed').innerText = message['linear_speed']
})

socket.on('next', (message) => {
  socket.emit('step', command())
})
