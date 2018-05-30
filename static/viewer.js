$.urlParam = function(name){
  var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
  if (results==null){
    return null;
  }
  else{
    return decodeURI(results[1]) || 0;
  }
}

var SCALE = 80
var DX = 450;
var DY = 500;

window.onload = function() {

  track = $.urlParam('track');
  capture = $.urlParam('capture');

  if (track === null) {
    window.alert('You must specify a track name as GET parameter "track"');
    return;
  }
  $.get("/track/" + track + "/path", function(data) {
    var t = document.getElementById("track");
    var ctxTrack = t.getContext("2d");

    landmarks = [0, 108, 142, 182, 250, 347]

    for (s in data) {
      for (var p in data[s]) {
        if (s == "left" && landmarks.includes(parseInt(p))) {
          ctxTrack.fillStyle="#FF0000"
          ctxTrack.fillRect(
            Math.trunc(SCALE * data[s][p][0]) + DX - 2,
            Math.trunc(SCALE * -data[s][p][2]) + DY - 2,
            4,4
          );
          console.log("LANDMARK " + p + " " + data[s][p][0].toFixed(2) + "," + data[s][p][2].toFixed(2))
        } else {
          ctxTrack.fillStyle="#000000";
          ctxTrack.fillRect(
            Math.trunc(SCALE * data[s][p][0]) + DX,
            Math.trunc(SCALE * -data[s][p][2]) + DY,
            1,1
          );
        }
      }
    }
  })

  if (capture !== null) {

    $.get("/track/" + track + "/capture/" + capture + "/integrated", function(data) {
      var t = document.getElementById("track");
      var ctxTrack = t.getContext("2d");

      console.log(data)

      for (var p in data) {
        ctxTrack.fillStyle="#999999";
        ctxTrack.fillRect(
          Math.trunc(SCALE * data[p][0]) + DX,
          Math.trunc(SCALE * data[p][2]) + DY,
          1,1
        );
      }
    })

    $.get("/track/" + track + "/capture/" + capture + "/annotated", function(data) {
      var t = document.getElementById("track");
      var ctxTrack = t.getContext("2d");

      for (var p in data) {
        ctxTrack.fillStyle="#FF0000";
        ctxTrack.fillRect(
          Math.trunc(SCALE * data[p][0]) + DX - 2,
          Math.trunc(SCALE * data[p][2]) + DY - 2,
          4,4
        );
      }
    })

    $.get("/track/" + track + "/capture/" + capture + "/corrected", function(data) {
      var t = document.getElementById("track");
      var ctxTrack = t.getContext("2d");

      for (var p in data) {
        ctxTrack.fillStyle="#0000FF";
        ctxTrack.fillRect(
          Math.trunc(SCALE * data[p][0]) + DX,
          Math.trunc(SCALE * data[p][2]) + DY,
          1,1
        );
      }
    })

    $.get("/track/" + track + "/capture/" + capture + "/reference", function(data) {
      var t = document.getElementById("track");
      var ctxTrack = t.getContext("2d");

      for (var p in data) {
        ctxTrack.fillStyle="#00FF00";
        ctxTrack.fillRect(
          Math.trunc(SCALE * data[p][0]) + DX,
          Math.trunc(SCALE * data[p][2]) + DY,
          1,1
        );
      }
    })
  }
};
