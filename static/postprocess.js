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
