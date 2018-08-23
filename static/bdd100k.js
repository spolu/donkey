PAGE_SIZE = 100

_data = []
_page = 0

var page_refresh = function() {
  $('#data').empty()

  $('#total').text(_data.length)
  $('#visible').text('0-0');

  if (_data.length === 0) {
    return;
  }

  for (var i = 0; i < PAGE_SIZE && PAGE_SIZE * _page + i < _data.length; i ++) {
    r = _data[PAGE_SIZE*_page + i]
    row = $('<tr>' +
      '<td>'+r[0]+'</td>'+
      '<td>'+r[1]+'</td>'+
      '<td>'+r[2]+'</td>'+
      '<td>'+r[3]+'</td>'+
      '<td>'+r[4]+'</td>'+
      '<td>'+
      '  <span data-name='+r[0]+' class="active true">y</span>'+
      '  <span data-name='+r[0]+' class="active false">n</span>'+
      '</td>'+
      '<td><span class="view">view</span></td>' +
      '<td><span class="objects">objects</span></td>'
    );

    (function(row, r) {
      t = document.getElementById("annotation-canvas");
      row.mouseover(function() {
        url = '/images/' + r[0] + '.jpg';
        $('#image').css('background-image', 'url(' + url + ')');
      });
      row.mouseout(function() {
        $('#image').css('background-image', 'none');
      });
      row.find('.objects').mouseover(function() {
        $.get("/videos/" + r[0] + '/objects', function(data) {
          ctx = t.getContext("2d");
          // console.log(data[0]);
          for (i in data) {
            console.log(data[i])
            ctx.fillStyle="red";
            ctx.fillRect(
              Math.floor(data[i].box2d['x1'] / 2),
              Math.floor(data[i].box2d['y1'] / 2),
              Math.floor((data[i].box2d['x2'] - data[i].box2d['x1']) / 2),
              Math.floor((data[i].box2d['y2'] - data[i].box2d['y1']) / 2),
            );
          }
        })
      });
      row.find('.objects').mouseout(function() {
        ctx = t.getContext("2d");
        ctx.clearRect(0, 0, t.width, t.height)
      });
    })(row, r);
    row.appendTo('#data');

    (function(row) {
      if (r[5] == 'true') {
        row.find('.active.true').addClass('selected')
      } else {
        row.find('.active.false').addClass('selected')
      }

      row.find('.active.true').click(function() {
        if (row.find('.active.false').hasClass('selected')) {
          row.find('.active.true').toggleClass('selected');
          row.find('.active.false').toggleClass('selected');
          $.ajax({
            type: 'POST',
            url: '/videos/'+row.find('.active.false').attr('data-name'),
            contentType: 'application/json',
            data: JSON.stringify({ active: 'true' }),
          });
        }
      });
      row.find('.active.false').click(function() {
        if (row.find('.active.true').hasClass('selected')) {
          row.find('.active.true').toggleClass('selected');
          row.find('.active.false').toggleClass('selected');
          $.ajax({
            type: 'POST',
            url: '/videos/'+row.find('.active.false').attr('data-name'),
            contentType: 'application/json',
            data: JSON.stringify({ active: 'false' }),
          });
        }
      })
    })(row);
  }

  $('#visible').text(PAGE_SIZE*_page + '-' + (PAGE_SIZE*(_page+1)-1))
}

var data_refresh = function() {
  query = '?'
  $('.selector.selected').each(function(i, el) {
    if (i > 0) {
      query += '&';
    }
    query += $(el).attr('data-attribute') + '[]=' + $(el).attr('data-value');
  });

  if (query === '?') {
    _data = [];
    _page = 0;
    page_refresh();
  } else {
    $.get("/videos" + query, function(data) {
      _data = data;
      _page = 0;
      page_refresh();
    })
  }
}

window.onload = function() {
  $.get("/attributes", function(data) {
    for (attr in data) {
      for (i in data[attr]) {
        selector = $('<span class="selector">' + data[attr][i] + '</span>');
        selector.attr('data-attribute', attr);
        selector.attr('data-value', data[attr][i]);
        (function(selector) {
          selector.click(function() {
            selector.toggleClass('selected');
            data_refresh()
          });
        })(selector);
        selector.appendTo('#' + attr + ' .selectors');
      }
    }
  });

  $(document).keypress(function(e) {
    if (e.which === 110) {
      _page += 1;
      if (_page * PAGE_SIZE > _data.length) {
        _page -= 1;
      }
      page_refresh();
    }
    if (e.which === 112) {
      _page -= 1;
      if (_page < 0) {
        _page = 0;
      }
      page_refresh();
    }
  })
}
