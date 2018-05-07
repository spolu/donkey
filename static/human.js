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

  var c = document.getElementById("camera");
  var ctx = c.getContext("2d");
  var imgData = ctx.createImageData(160,120);
  for (var w = 0; w < 120; w++) {
    for (var h = 0; h < 160; h++) {
      imgData.data[((w * (160 * 4)) + (h * 4)) + 0] = Math.floor(
        (message['camera'][w][h] + 1) * 127.5
      )
      imgData.data[((w * (160 * 4)) + (h * 4)) + 1] = Math.floor(
        (message['camera'][w][h] + 1) * 127.5
      )
      imgData.data[((w * (160 * 4)) + (h * 4)) + 2] = Math.floor(
        (message['camera'][w][h] + 1) * 127.5
      )
      imgData.data[((w * (160 * 4)) + (h * 4)) + 3] = 255
    }
  }
  ctx.putImageData(imgData,0,0);
})

socket.on('next', (message) => {
  socket.emit('step', command())
})
