var capture = null;

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
  capture = $.urlParam('capture');

  if (capture !== null) {
    $('#index').change(function() {
      url = "/capture/" + capture + "/camera/" + $('#index').val() + ".jpeg" +
        "?" + (new Date()).getTime();
      $('#camera').css("background-image", "url(" + url + ")");
    })
  }
};
