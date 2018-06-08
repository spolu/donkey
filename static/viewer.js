var LANDMARKS = [0, 55, 122, 184, 230, 300, 347]
var LANDMARKS_VALUE = {}

var SCALE = 80
var DX = 450;
var DY = 450;

var track = null;
var capture = null;
var max = null
var min = null

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

var integrated_refresh = function() {
  $.get("/track/" + track + "/capture/" + capture + "/integrated", function(data) {
    for (var p in data) {
      if (max !== null && p >= max) {
        break;
      }
      if (min !== null && p <= min) {
        continue
      }
      ctxTrack.fillStyle="#999999";
      ctxTrack.fillRect(
        Math.trunc(SCALE * data[p][0]) + DX,
        Math.trunc(SCALE * -data[p][2]) + DY,
        1,1
      );
    }
  })
}

var annotated_refresh = function() {
  $.get("/track/" + track + "/capture/" + capture + "/annotated", function(data) {
    for (var p in data['annotated']) {
      if (max !== null && data['indices'][p] >= max) {
        break;
      }
      if (min !== null && data['indices'][p] <= min) {
        continue
      }
      ctxTrack.fillStyle="#00FF00";
      ctxTrack.fillRect(
        Math.trunc(SCALE * data['annotated'][p][0]) + DX - 3,
        Math.trunc(SCALE * -data['annotated'][p][2]) + DY - 3,
        6,6
      );
    }
  })
}

var reference_refresh = function() {
  $.get("/track/" + track + "/capture/" + capture + "/reference", function(data) {
    for (var p in data) {
      ctxTrack.fillStyle="#00FF00";
      ctxTrack.fillRect(
        Math.trunc(SCALE * data[p][0]) + DX,
        Math.trunc(SCALE * -data[p][2]) + DY,
        1,1
      );
    }
  })
}

var inferred_refresh = function() {
  $.get("/track/" + track + "/capture/" + capture + "/inferred", function(data) {
    for (var p in data) {
      ctxTrack.fillStyle="#00FF00";
      ctxTrack.fillRect(
        Math.trunc(SCALE * data[p][0]) + DX - 2,
        Math.trunc(SCALE * -data[p][2]) + DY - 2,
        4,4
      );
    }
  })
}

var corrected_refresh = function() {
  $.get("/track/" + track + "/capture/" + capture + "/corrected", function(data) {
    for (var p in data) {
      if (max !== null && p >= max) {
        break;
      }
      if (min !== null && p <= min) {
        continue
      }
      ctxTrack.fillStyle="#0000FF";
      ctxTrack.fillRect(
        Math.trunc(SCALE * data[p][0]) + DX - 1,
        Math.trunc(SCALE * -data[p][2]) + DY - 1,
        2,2
      );
    }
  })
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

window.onload = function() {

  t = document.getElementById("track");
  ctxTrack = t.getContext("2d");

  track = $.urlParam('track');
  capture = $.urlParam('capture');
  if ($.urlParam('max') != null) {
    max = parseInt($.urlParam('max'))
  }
  if ($.urlParam('min') != null) {
    min = parseInt($.urlParam('min'))
  }

  if (track === null) {
    window.alert('You must specify a track name as GET parameter "track"');
    return;
  }

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

  if (capture !== null) {

    annotated_refresh()
    reference_refresh()
    inferred_refresh()
    corrected_refresh()
    integrated_refresh()

    $('#index').change(function() {
      url = "/track/" + track + "/capture/" + capture + "/camera/" + $('#index').val() + ".jpeg"
      $('#camera').css("background-image", "url(" + url + ")");
    })
  }
};
