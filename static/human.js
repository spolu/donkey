var socket = io.connect("ws://127.0.0.1:9091")

var SCALE = 40
var DX = 450;
var DY = 450;
var INPUT_WIDTH = 160
var INPUT_HEIGHT = 120

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
    c['throttle'] = 0.75
  }
  if (keypressed['a']) {
    c['steering'] = -1.0
  }
  if (keypressed['d']) {
    c['steering'] = 1.0
  }
  if (keypressed['s']) {
    c['brake'] = 0.0
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
  document.getElementById('time').innerText = message['observation']['time']
  document.getElementById('linear_speed').innerText = message['observation']['track_linear_speed']
  document.getElementById('position').innerText = message['observation']['position'].toString()

  if (message['observation']['camera']) {
    camera = message['observation']['camera']
    var c = document.getElementById("camera");
    var ctxCamera = c.getContext("2d");
    var imgData = ctxCamera.createImageData(INPUT_WIDTH,INPUT_HEIGHT);
    for (var w = 0; w < camera.length; w++) {
      for (var h = 0; h < camera[w].length; h++) {
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 0] = Math.floor(
          (camera[w][h] + 1)
        )
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 1] = Math.floor(
          (camera[w][h] + 1)
        )
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 2] = Math.floor(
          (camera[w][h] + 1)
        )
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 3] = 255
      }
    }
    ctxCamera.putImageData(imgData,0,0);
  }

  if (message['observation']['generated']) {
    generated = message['observation']['generated']
    var c = document.getElementById("generated");
    var ctxCamera = c.getContext("2d");
    var imgData = ctxCamera.createImageData(INPUT_WIDTH,INPUT_HEIGHT);
    for (var w = 0; w < generated.length; w++) {
      for (var h = 0; h < generated[w].length; h++) {
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 0] = Math.floor(
          (generated[w][h] + 1)
        )
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 1] = Math.floor(
          (generated[w][h] + 1)
        )
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 2] = Math.floor(
          (generated[w][h] + 1)
        )
        imgData.data[((w * (INPUT_WIDTH * 4)) + (h * 4)) + 3] = 255
      }
    }
    ctxCamera.putImageData(imgData,0,0);
  }

  var t = document.getElementById("track");
  var ctxTrack = t.getContext("2d");

  ctxTrack.fillStyle="#000000";
  ctxTrack.fillRect(
    Math.trunc(SCALE * message['observation']['position'][0]) + DX,
    Math.trunc(SCALE * message['observation']['position'][2]) + DY,
    1,1
  );

  if (message['model']) {
    ctxTrack.fillStyle="#FF0000";
    ctxTrack.fillRect(
      Math.trunc(SCALE * message['model']['position'][0]) + DX,
      Math.trunc(SCALE * message['model']['position'][2]) + DY,
      1,1
    );
  }
})

socket.on('next', (message) => {
  socket.emit('step', command())
})
