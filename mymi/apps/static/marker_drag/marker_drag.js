// ── Streamlit postMessage helpers ──────────────────────────────────────────
function _post(m) {
  var msg = Object.assign({ isStreamlitMessage: true }, m);
  try { window.parent.postMessage(msg, '*'); } catch (e) {}
  try { if (window.top !== window.parent) window.top.postMessage(msg, '*'); } catch (e) {}
}
function setVal(v) { _post({ type: 'streamlit:setComponentValue', dataType: 'json', value: v }); }
function setH(h)   { _post({ type: 'streamlit:setFrameHeight', height: h }); }

// Signal ready immediately so Streamlit doesn't time out waiting for the handshake.
_post({ type: 'streamlit:componentReady', apiVersion: 1 });
setH(1);  // non-zero initial height so Streamlit registers the frame

// ── State ──────────────────────────────────────────────────────────────────
var canvas    = document.getElementById('c');
var ctx       = canvas.getContext('2d');
var statusEl  = document.getElementById('status');
var img       = new Image();
var args      = null;

// markers: array of { label, x, y } in *image* pixel coords
var markers   = [];
var dragging  = false;
var dragIdx   = -1;
var selectedIdx = -1;   // used in 'arrow' mode - marker selected for keyboard nudging.
var swapIdx   = -1;     // used in 'swap' mode - marker selected awaiting a target index to swap with.
var lastImgSrc = '';

var swapPopup = document.getElementById('swap-popup');
var swapInput = document.getElementById('swap-input');
var swapLabel = document.getElementById('swap-popup-label');

var HIT_PX    = 14;  // hit-test radius in canvas pixels
var DOT_R     = 7;   // drawn dot radius in canvas pixels
var DOT_BORDER = 1.5;

function getMode() {
  return (args && args.marker_mode) || 'arrow';
}

// ── Coordinate helpers ─────────────────────────────────────────────────────
function imgToCanvas(ix, iy) {
  if (!args) return [0, 0];
  return [
    ix * (canvas.width  / args.img_w),
    iy * (canvas.height / args.img_h)
  ];
}

function canvasToImg(cx, cy) {
  if (!args) return [0, 0];
  return [
    Math.max(0, Math.min(args.img_w - 1, cx * (args.img_w / canvas.width))),
    Math.max(0, Math.min(args.img_h - 1, cy * (args.img_h / canvas.height)))
  ];
}

function hitTest(cx, cy) {
  for (var i = 0; i < markers.length; i++) {
    var m = markers[i];
    if (isNaN(m.x) || isNaN(m.y)) continue;
    var pos = imgToCanvas(m.x, m.y);
    var dx = cx - pos[0], dy = cy - pos[1];
    if (dx * dx + dy * dy <= HIT_PX * HIT_PX) return i;
  }
  return -1;
}

// ── Drawing ────────────────────────────────────────────────────────────────
function draw() {
  if (!img.complete || !args) return;

  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  var style     = args.marker_style || 'cross';
  var thickness = (args.marker_thickness != null) ? args.marker_thickness : 2.5;
  var opacity   = (args.marker_opacity != null) ? args.marker_opacity : 0.5;

  for (var i = 0; i < markers.length; i++) {
    var m = markers[i];
    if (isNaN(m.x) || isNaN(m.y)) continue;

    var pos = imgToCanvas(m.x, m.y);
    var cx = pos[0], cy = pos[1];
    var isActive = (i === dragIdx && dragging) || (i === selectedIdx && getMode() === 'arrow') || (i === swapIdx && getMode() === 'swap');
    var col = isActive ? '#00ffff' : '#ffff00';
    var labelOffset;

    ctx.save();
    ctx.globalAlpha = opacity;

    if (style === 'point') {
      var r = Math.max(2, thickness * 2.5);
      labelOffset = r + 2;

      // Shadow, for contrast against bright backgrounds.
      ctx.beginPath();
      ctx.arc(cx, cy, r + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
    } else {
      var ARM = DOT_R;
      labelOffset = ARM + 2;

      // Thin black shadow for contrast, drawn wider and underneath.
      ctx.strokeStyle = 'rgba(0,0,0,0.6)';
      ctx.lineWidth   = thickness + 2;
      ctx.lineCap     = 'round';
      ctx.beginPath();
      ctx.moveTo(cx - ARM, cy); ctx.lineTo(cx + ARM, cy);
      ctx.moveTo(cx, cy - ARM); ctx.lineTo(cx, cy + ARM);
      ctx.stroke();

      // Crosshair.
      ctx.strokeStyle = col;
      ctx.lineWidth   = thickness;
      ctx.beginPath();
      ctx.moveTo(cx - ARM, cy); ctx.lineTo(cx + ARM, cy);
      ctx.moveTo(cx, cy - ARM); ctx.lineTo(cx, cy + ARM);
      ctx.stroke();
    }
    ctx.restore();

    // Label offset to top-right of the marker - drawn at full opacity for readability.
    ctx.font         = 'bold 10px sans-serif';
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle    = 'rgba(0,0,0,0.7)';
    ctx.fillText(m.label, cx + labelOffset, cy - 1);
    ctx.fillStyle = col;
    ctx.fillText(m.label, cx + labelOffset - 1, cy - 2);
  }
}

// ── Swap mode (mode 3) ──────────────────────────────────────────────────────
// Click a marker, then type another marker's index + Enter in the popup to swap their positions.
function openSwapPopup(idx, cx, cy) {
  swapIdx = idx;
  swapPopup.style.left = (cx + 10) + 'px';
  swapPopup.style.top  = (cy + 10) + 'px';
  swapPopup.style.display = 'block';
  swapLabel.textContent = 'Swap marker ' + markers[idx].label + ' with #:';
  swapInput.value = '';
  // Deferred so it runs after the mousedown's default focus-shift behaviour (focusing
  // synchronously here gets immediately stolen back by the canvas otherwise).
  setTimeout(function () { swapInput.focus(); }, 0);
  draw();
  setStatus('Marker ' + markers[idx].label + ' selected - type another marker index and press Enter to swap.');
}

function closeSwapPopup() {
  swapIdx = -1;
  swapPopup.style.display = 'none';
  draw();
}

function doSwap(targetIdx) {
  if (targetIdx < 0 || targetIdx >= markers.length || targetIdx === swapIdx) return;
  var a = markers[swapIdx], b = markers[targetIdx];
  var ax = a.x, ay = a.y;
  a.x = b.x; a.y = b.y;
  b.x = ax; b.y = ay;
  setStatus('Swapped markers ' + a.label + ' and ' + b.label + '.');
  draw();
  sendPositions();
}

swapInput.addEventListener('keydown', function (e) {
  e.stopPropagation();   // don't let digits trigger the arrow-key nudge listener below.
  if (e.key === 'Escape') {
    closeSwapPopup();
    e.preventDefault();
  } else if (e.key === 'Enter') {
    var v = parseInt(swapInput.value, 10);
    if (!isNaN(v)) doSwap(v);
    closeSwapPopup();
    e.preventDefault();
  }
});

// Dismiss the popup when clicking anywhere outside it.
document.addEventListener('mousedown', function (e) {
  if (swapPopup.style.display === 'block' && e.target !== swapInput && !swapPopup.contains(e.target)) {
    closeSwapPopup();
  }
});

// ── Mouse events ───────────────────────────────────────────────────────────
function getCanvasPos(e) {
  var rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

canvas.addEventListener('mousedown', function (e) {
  var pos = getCanvasPos(e);
  var hit = hitTest(pos[0], pos[1]);

  if (getMode() === 'drag') {
    if (hit >= 0) {
      dragging = true;
      dragIdx  = hit;
      canvas.style.cursor = 'grabbing';
      setStatus('Dragging marker ' + markers[hit].label + '...');
      e.preventDefault();
    }
    return;
  }

  if (getMode() === 'swap') {
    e.preventDefault();    // stop the canvas click from stealing focus from the popup input.
    e.stopPropagation();   // don't let this same click reach the document dismiss-listener below.
    if (hit >= 0) {
      openSwapPopup(hit, pos[0], pos[1]);
    } else {
      closeSwapPopup();
    }
    return;
  }

  // Arrow-key mode: click selects a marker (no drag) - arrow keys then nudge it.
  selectedIdx = hit;
  setStatus(hit >= 0
    ? 'Selected marker ' + markers[hit].label + ' - use arrow keys to move (step=' + (args.arrow_step || 0.5) + 'px).'
    : 'No marker selected - click a marker to select it.');
  draw();
});

canvas.addEventListener('mousemove', function (e) {
  var pos = getCanvasPos(e);
  var hit = hitTest(pos[0], pos[1]);

  if (getMode() !== 'drag') {
    canvas.style.cursor = hit >= 0 ? 'pointer' : 'default';
    return;
  }

  if (dragging) {
    var ip = canvasToImg(pos[0], pos[1]);
    markers[dragIdx].x = ip[0];
    markers[dragIdx].y = ip[1];
    setStatus('Marker ' + markers[dragIdx].label + ': x=' + ip[0].toFixed(1) + ' y=' + ip[1].toFixed(1));
    draw();
  } else {
    canvas.style.cursor = hit >= 0 ? 'grab' : 'crosshair';
  }
});

function finishDrag() {
  if (!dragging) return;
  dragging = false;
  canvas.style.cursor = 'crosshair';
  setStatus('Marker ' + markers[dragIdx].label + ' placed. Release to confirm.');
  dragIdx = -1;
  draw();
  sendPositions();
}

canvas.addEventListener('mouseup',    finishDrag);
canvas.addEventListener('mouseleave', finishDrag);
// Also catch mouseup outside the iframe
document.addEventListener('mouseup',  finishDrag);

// ── Keyboard nudging (mode 1) ───────────────────────────────────────────────
document.addEventListener('keydown', function (e) {
  if (getMode() !== 'arrow' || selectedIdx < 0 || selectedIdx >= markers.length) return;
  var m = markers[selectedIdx];
  if (isNaN(m.x) || isNaN(m.y)) return;

  var step = (args && args.arrow_step != null) ? args.arrow_step : 0.5;
  var dx = 0, dy = 0;
  if (e.key === 'ArrowUp')         dy = -step;
  else if (e.key === 'ArrowDown')  dy = step;
  else if (e.key === 'ArrowLeft')  dx = -step;
  else if (e.key === 'ArrowRight') dx = step;
  else return;
  e.preventDefault();

  m.x = Math.max(0, Math.min(args.img_w - 1, m.x + dx));
  m.y = Math.max(0, Math.min(args.img_h - 1, m.y + dy));
  setStatus('Marker ' + m.label + ': x=' + m.x.toFixed(1) + ' y=' + m.y.toFixed(1));
  draw();
  sendPositions();
});

// ── Communication ──────────────────────────────────────────────────────────
function sendPositions() {
  var out = markers.map(function (m) {
    return { label: m.label, x: m.x, y: m.y };
  });
  setVal({
    markers: out,
    frame_idx: args ? args.frame_idx : -1,
    crop_x0: args ? args.crop_x0 : 0,
    crop_y0: args ? args.crop_y0 : 0
  });
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

// ── Responsive canvas sizing ───────────────────────────────────────────────
// Use the iframe's actual container width so the canvas fills the column.
function sizeCanvas() {
  if (!args) return false;
  var wrap = document.getElementById('canvas-wrap');
  var w = (wrap && wrap.clientWidth > 0) ? wrap.clientWidth : (args.canvas_w || 580);
  var h = Math.max(1, Math.round(args.img_h * w / args.img_w));
  if (canvas.width === w && canvas.height === h) return false;
  canvas.width  = w;
  canvas.height = h;
  setH(h);
  return true;
}

window.addEventListener('resize', function () { if (sizeCanvas()) draw(); });

// ── Streamlit render handler ───────────────────────────────────────────────
img.onload = function () { sizeCanvas(); draw(); };

window.addEventListener('message', function (e) {
  if (!e.data || e.data.type !== 'streamlit:render') return;
  args = e.data.args;
  closeSwapPopup();

  sizeCanvas();

  // Update markers only when not mid-drag
  if (!dragging) {
    markers = (args.markers || []).map(function (m) {
      return { label: m.label, x: m.x, y: m.y };
    });
  }

  // Reload image only if it changed
  var newSrc = args.img_src || '';
  if (newSrc !== lastImgSrc) {
    lastImgSrc = newSrc;
    img.src = newSrc;              // onload -> sizeCanvas + draw()
  } else {
    draw();                        // same frame, just redraw markers
  }

  setStatus('Frame ' + (args.frame_idx + 1) + ' | ' + markers.length + ' markers | ' +
    (getMode() === 'drag' ? 'drag a yellow dot to reposition'
      : getMode() === 'swap' ? 'click a marker, then type another index + Enter to swap'
      : 'click a marker, then use arrow keys to nudge it'));
});
