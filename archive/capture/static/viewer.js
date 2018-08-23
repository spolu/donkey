var LANDMARKS = [0, 55, 122, 184, 230, 300, 347]
var LANDMARKS_VALUE = {}

var SCALE = 80
var DX = 450;
var DY = 450;

var track = null;
var capture = null;
var min = 0
var delta = 9999

var t = null;
var ctxTrack = null;

$.urlParam = function(name){
  var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
  if (results==null){
    return null;
  }
  else{
    return decodeURI(results[1]) || 0;
  }
}

var data_refresh = function(type, color, size) {
  $.get("/track/" + track + "/capture/" + capture + "/" + type, function(data) {
    for (var p in data[type]) {
      if (data['indices'][p] >= min + delta) {
        break;
      }
      if(data['indices'][p] <= min) {
        continue
      }
      ctxTrack.fillStyle=color;
      ctxTrack.fillRect(
        Math.trunc(SCALE * data[type][p][0]) + DX - size,
        Math.trunc(SCALE * -data[type][p][2]) + DY - size,
        2*size,2*size
      );
    }
  })
}

var inferred_refresh = function() {
  data_refresh('inferred', 'orange', 2)
}

var annotated_refresh = function() {
  data_refresh('annotated', '#00FF00', 3)
}

var corrected_refresh = function() {
  data_refresh('corrected', '#0000FF', 1)
}

var integrated_refresh = function() {
  data_refresh('integrated', '#888', 1)
}

var landmark_refresh = function(l, color) {
  ctxTrack.fillStyle = color
  ctxTrack.fillRect(
    Math.trunc(SCALE * LANDMARKS_VALUE[l][0]) + DX - 3,
    Math.trunc(SCALE * -LANDMARKS_VALUE[l][2]) + DY - 3,
    6,6
  );
}

var landmarks_refresh = function() {
  $('#landmarks').empty()
  LANDMARKS.forEach(function(l) {
    landmark = jQuery(
      '<div>' +
        '' + l + ' - ' +
        LANDMARKS_VALUE[l][0].toFixed(2) + ',' + LANDMARKS_VALUE[l][2].toFixed(2) +
      '</div>', {
      id: 'l' + l,
      class: 'landmark',
      }).css({
        'margin-top': '5px',
        'cursor': 'pointer',
      })
    landmark.mouseover(function() {
      landmark_refresh(l, "#000000")
    })
    landmark.mouseout(function() {
      landmark_refresh(l, "#FF0000")
    })
    landmark.click(function() {
      index = $('#index').val()
      url = "/track/" + track + "/capture/" + capture + "/annotate/" + index + "/landmark/" + l
      $.get(url, function(data) {
        annotated_refresh()
      })
    })
    landmark.appendTo('#landmarks');
  })
}

var track_refresh = function() {
  $.get("/track/" + track + "/path", function(data) {
    for (s in data) {
      for (var p in data[s]) {
        if (s == "center" && LANDMARKS.includes(parseInt(p))) {
          LANDMARKS_VALUE[parseInt(p)] = [
            data[s][p][0],
            data[s][p][1],
            data[s][p][2],
          ]
          landmark_refresh(parseInt(p), "#FF0000")
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
    landmarks_refresh()
  })
}

var all_refresh = function(track) {
  ctxTrack.fillStyle='white';
  ctxTrack.fillRect(
    0, 0, 800, 800
  );

  if (track) {
    track_refresh()
  }
  annotated_refresh()
  inferred_refresh()
  corrected_refresh()
  integrated_refresh()
}

window.onload = function() {

  t = document.getElementById("track");
  ctxTrack = t.getContext("2d");

  track = $.urlParam('track');
  capture = $.urlParam('capture');

  if (track === null) {
    window.alert('You must specify a track name as GET parameter "track"');
    return;
  }

  track_refresh()

  if (capture !== null) {
    all_refresh(false)

    $('#index').change(function() {
      url = "/track/" + track + "/capture/" + capture + "/camera/" + $('#index').val() + ".jpeg" +
        "?" + (new Date()).getTime();
      $('#camera').css("background-image", "url(" + url + ")");
    })

    $('#min').change(function() {
      min = parseInt($('#min').val())
      all_refresh(true)
    })
    $('#delta').change(function() {
      delta = parseInt($('#delta').val())
      all_refresh(true)
    })
  }
};
