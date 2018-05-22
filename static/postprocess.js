$.get("/track/newworld", function(data) {
  var t = document.getElementById("track");
  var ctxTrack = t.getContext("2d");

  for (s in data) {
    for (var p in data[s]) {
      ctxTrack.fillStyle="#000000";
      ctxTrack.fillRect(
        Math.trunc(3 * data[s][p][0]) + 350,
        Math.trunc(3 * data[s][p][2]) + 400,
        1,1
      );
    }
  }
})

$.get("/capture/newworld/5.5", function(data) {
  var t = document.getElementById("track");
  var ctxTrack = t.getContext("2d");

  for (var p in data) {
    ctxTrack.fillStyle="#FF0000";
    ctxTrack.fillRect(
      Math.trunc(3 * data[p][0]) + 350,
      Math.trunc(3 * data[p][2]) + 400,
      1,1
    );
  }
})

$.get("/reference/newworld", function(data) {
  var t = document.getElementById("track");
  var ctxTrack = t.getContext("2d");

  for (var p in data) {
    ctxTrack.fillStyle="#00FF00";
    ctxTrack.fillRect(
      Math.trunc(3 * data[p][0]) + 350,
      Math.trunc(3 * data[p][2]) + 400,
      1,1
    );
  }
})
