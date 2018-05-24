$.urlParam = function(name){
  var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
  if (results==null){
    return null;
  }
  else{
    return decodeURI(results[1]) || 0;
  }
}

window.onload = function() {

  track = $.urlParam('track');
  capture = $.urlParam('capture');

  if (track === null) {
    window.alert('You must specify a track name as GET parameter "track"');
    return;
  }
  if (capture === null) {
    window.alert('You must specify a capture name as GET parameter "capture"');
    return;
  }

  $.get("/track/" + track + "/path", function(data) {
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

  $.get("/track/" + track + "/capture/" + capture + "/annotated", function(data) {
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

  $.get("/track/" + track + "/capture/" + capture + "/reference", function(data) {
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
};
