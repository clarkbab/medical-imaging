function _post(m){
  var msg=Object.assign({isStreamlitMessage:true},m);
  try{window.parent.postMessage(msg,'*');}catch(e){}
  try{if(window.top!==window.parent)window.top.postMessage(msg,'*');}catch(e){}
}
function setVal(v){_post({type:'streamlit:setComponentValue',dataType:'json',value:v});}
function setH(h){_post({type:'streamlit:setFrameHeight',height:h});}
// Track the last ox/oy we sent so we can ignore stale Python args while a drag is in-flight.
var _pendingOx=null,_pendingOy=null;
// force_counter: when Python increments this, JS unconditionally accepts the new offset.
var _forceCounter=0;
function _sendPos(){_pendingOx=ox;_pendingOy=oy;setVal({offset_x:ox,offset_y:oy,zoom:{x0:zoom.x0,y0:zoom.y0,x1:zoom.x1,y1:zoom.y1}});}

var YAXIS_W=38, XAXIS_H=20;
var args=null,ox=0,oy=0,drag=false,mx0=0,my0=0,ox0=0,oy0=0,_rendered=false;
var img1=new Image(),img2=new Image(),rdy=0;
var c1=document.getElementById('c1'),c2=document.getElementById('c2');
var ya1=document.getElementById('ya1'),ya2=document.getElementById('ya2');
var xa1=document.getElementById('xa1'),xa2=document.getElementById('xa2');
var g1=c1.getContext('2d'),g2=c2.getContext('2d');

// Zoom: normalised image coords of the visible rectangle [0,1].
var zoom={x0:0,y0:0,x1:1,y1:1};
// Mode: 'pan' moves the contour overlay; 'box' draws a selection to zoom into.
var mode='pan';
var boxStart=null,boxCur=null;
var _zoomTimer=null;

function setMode(m){
  mode=m;
  try{window.parent._contourMode=m;}catch(ex){}
  document.getElementById('btn-pan').className=m==='pan'?'active':'';
  document.getElementById('btn-box').className=m==='box'?'active':'';
  document.getElementById('btn-deform').className=m==='deform'?'active':'';
  var cursor=m==='pan'?'grab':m==='deform'?'default':'crosshair';
  [c1,c2].forEach(function(c){c.style.cursor=cursor;});
}
document.getElementById('btn-pan').onclick=function(){setMode('pan');};
document.getElementById('btn-box').onclick=function(){setMode('box');};
document.getElementById('btn-deform').onclick=function(){setMode('deform');};

// Internal keydown: hotkeys and contour nudge when focus is inside the iframe.
document.addEventListener('keydown',function(e){
  if(e.repeat)return;
  var tag=e.target?e.target.tagName:'';
  if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT')return;
  var k=e.key;
  // Shift+Arrow: navigate projection index by clicking the prev/next buttons.
  if(e.shiftKey&&(k==='ArrowLeft'||k==='ArrowRight')){
    e.preventDefault();
    var lbl=k==='ArrowLeft'?'◀':'▶';
    try{window.parent.document.querySelectorAll('button').forEach(function(b){
      if(b.textContent.trim()===lbl)b.click();
    });}catch(ex){}
    return;
  }
  if(k==='m'||k==='M')setMode('pan');
  else if(k==='z'||k==='Z')setMode('box');
  else if(k==='d'||k==='D')setMode('deform');
});

// Global hotkeys from parent page — attached in CAPTURE phase so we intercept before any
// focused element (slider, button, etc.) ever sees the event.
if(!window.parent._contourKeyCapture){
  window.parent._contourKeyCapture=true;
  window.parent.document.addEventListener('keydown',function(e){
    if(e.repeat)return;
    var k=e.key;
    // Shift+Arrow: projection navigation, regardless of focus.
    if(e.shiftKey&&(k==='ArrowLeft'||k==='ArrowRight')){
      e.preventDefault();
      e.stopImmediatePropagation();
      var lbl=k==='ArrowLeft'?'◀':'▶';
      window.parent.document.querySelectorAll('button').forEach(function(b){
        if(b.textContent.trim()===lbl)b.click();
      });
      return;
    }
    // Mode hotkeys: only fire when focus is not in a text input.
    var tag=e.target?e.target.tagName:'';
    if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT')return;
    var _mode=k==='m'||k==='M'?'pan':k==='z'||k==='Z'?'box':k==='d'||k==='D'?'deform':null;
    if(_mode)document.querySelectorAll('iframe').forEach(function(fr){
      try{fr.contentWindow.setMode(_mode);}catch(ex){}
    });
  },true);  // true = capture phase
}

window.addEventListener('message',function(e){
  if(!e.data)return;
  if(e.data.type!=='streamlit:render')return;
  _rendered=true;
  args=e.data.args;
  var _ax=args.offset_x||0,_ay=args.offset_y||0;
  var _fc=args.force_counter||0;
  if(_fc!==_forceCounter){
    _forceCounter=_fc;ox=_ax;oy=_ay;_pendingOx=null;_pendingOy=null;
  } else if(_pendingOx===null){ox=_ax;oy=_ay;}
  else if(_ax===_pendingOx&&_ay===_pendingOy){_pendingOx=null;_pendingOy=null;}
  var z=args.zoom;
  if(z&&typeof z.x0==='number') zoom={x0:z.x0,y0:z.y0,x1:z.x1,y1:z.y1};
  document.getElementById('wrap').classList.toggle('prospective',!!args.prospective);
  document.getElementById('toolbar').classList.toggle('prospective',!!args.prospective);
  if(args.prospective&&(mode==='pan'||mode==='deform'))setMode('box');
  rdy=0; img1=new Image(); img2=new Image();
  function onload(){if(++rdy>=2)draw();}
  img1.onload=onload; img2.onload=onload;
  img1.src=args.treat_src; img2.src=args.drr_src;
});

function niceStep(range,n){
  var raw=range/n;
  var mag=Math.pow(10,Math.floor(Math.log10(raw)));
  var norm=raw/mag;
  return(norm<1.5?1:norm<3.5?2:norm<7.5?5:10)*mag;
}

// Draw Y axis ticks on a dedicated axis canvas.
// isLeft=true: ticks on right edge (pointing into the image); labels left-aligned.
// isLeft=false: ticks on left edge; labels right-aligned.
function drawYAxis(ac,ch,zy0,zh,ih,isLeft){
  var w=ac.width;
  var g=ac.getContext('2d');
  g.clearRect(0,0,w,ch);
  g.save();
  g.font='10px sans-serif';g.fillStyle='#555';g.strokeStyle='#aaa';g.lineWidth=1;
  var yPx0=zy0*ih,yPx1=(zy0+zh)*ih,yRange=yPx1-yPx0;
  if(yRange<=0){g.restore();return;}
  var yStep=100,TICK=4;
  var lblMin=Math.ceil((ih-yPx1)/yStep)*yStep;
  var lblMax=Math.floor((ih-yPx0)/yStep)*yStep;
  for(var lbl=lblMin;lbl<=lblMax;lbl+=yStep){
    var py=ih-lbl;
    var cy=(py-yPx0)/yRange*ch;
    g.beginPath();
    if(isLeft){g.moveTo(w-TICK,cy);g.lineTo(w,cy);}
    else{g.moveTo(0,cy);g.lineTo(TICK,cy);}
    g.stroke();
    g.textAlign=isLeft?'right':'left';g.textBaseline='middle';
    g.fillText(String(lbl),isLeft?w-TICK-2:TICK+2,cy);
  }
  g.restore();
}

// Draw X axis ticks on a dedicated axis canvas below the image.
function drawXAxis(ac,cw,zx0,zw,iw){
  var h=ac.height;
  var g=ac.getContext('2d');
  g.clearRect(0,0,cw,h);
  g.save();
  g.font='10px sans-serif';g.fillStyle='#555';g.strokeStyle='#aaa';g.lineWidth=1;
  var xPx0=zx0*iw,xPx1=(zx0+zw)*iw,xRange=xPx1-xPx0;
  if(xRange<=0){g.restore();return;}
  var xStep=100,TICK=4;
  for(var px=Math.ceil(xPx0/xStep)*xStep;px<=xPx1;px+=xStep){
    var cx=(px-xPx0)/xRange*cw;
    g.beginPath();g.moveTo(cx,0);g.lineTo(cx,TICK);g.stroke();
    g.textAlign='center';g.textBaseline='top';
    g.fillText(String(Math.round(px)),cx,TICK+1);
  }
  g.restore();
}

function draw(){
  if(!img1.complete||!img2.complete||!args)return;
  var wrap=document.getElementById('wrap');
  var tb=document.getElementById('toolbar');
  // In prospective mode only one panel is shown; use the full wrap width for it.
  var panelW=args.prospective
    ?Math.max(1,wrap.clientWidth-4)
    :Math.max(1,Math.floor((wrap.clientWidth-4)/2));
  var cw=Math.max(1,panelW-YAXIS_W);
  var iw=args.img_w,ih=args.img_h;
  var ch=Math.max(1,Math.round(ih/iw*cw));

  c1.width=cw;c1.height=ch;c2.width=cw;c2.height=ch;
  ya1.width=YAXIS_W;ya1.height=ch;ya2.width=YAXIS_W;ya2.height=ch;
  xa1.width=cw;xa2.width=cw;xa1.height=XAXIS_H;xa2.height=XAXIS_H;
  setH(ch+XAXIS_H+(tb?tb.offsetHeight+6:32));

  var zx0=zoom.x0,zy0=zoom.y0,zw=zoom.x1-zoom.x0,zh=zoom.y1-zoom.y0;
  var xs=args.cx,ys=args.cy;

  var cHex=args.contour_color||'#ff3c3c';
  var cR=parseInt(cHex.slice(1,3),16),cG=parseInt(cHex.slice(3,5),16),cB=parseInt(cHex.slice(5,7),16);
  var cStyle='rgba('+cR+','+cG+','+cB+',0.9)';
  var linestyle=(args.contour_linestyle||'Solid').toLowerCase();
  // Scale: canvas pixels per image pixel. Used to ensure solid dots always overlap.
  var _scaleX=cw/(zw*iw),_scaleY=ch/(zh*ih),_scale=Math.max(_scaleX,_scaleY);
  var ptSz=linestyle==='dotted'?1:linestyle==='solid'?Math.max(2,Math.ceil(_scale)+1):2;
  var ptSkip=linestyle==='dashed'?4:1;
  [[g1,img1],[g2,img2]].forEach(function(pair){
    var g=pair[0],img=pair[1];
    g.drawImage(img,zx0*iw,zy0*ih,zw*iw,zh*ih,0,0,cw,ch);
    g.fillStyle=cStyle;
    for(var i=0;i<xs.length;i+=ptSkip){
      // oy sign: positive oy shifts contour up (decreases canvas y).
      var nx=(xs[i]+ox)/iw, ny=(ys[i]-oy)/ih;
      var px=(nx-zx0)/zw*cw, py=(ny-zy0)/zh*ch;
      if(px>=0&&px<cw&&py>=0&&py<ch)g.fillRect(px|0,py|0,ptSz,ptSz);
    }
  });

  // Margin rectangle (prospective mode only).
  if(args.show_margin&&args.pixel_spacing>0){
    var mPx=args.margin_width_mm/args.pixel_spacing;
    var mnx0=mPx/iw,mnx1=(iw-mPx)/iw,mny0=mPx/ih,mny1=(ih-mPx)/ih;
    var mcx0=(mnx0-zx0)/zw*cw,mcx1=(mnx1-zx0)/zw*cw;
    var mcy0=(mny0-zy0)/zh*ch,mcy1=(mny1-zy0)/zh*ch;
    [g1,g2].forEach(function(g){
      g.save();
      g.strokeStyle='rgba(255,200,0,0.85)';g.lineWidth=1.5;g.setLineDash([6,4]);
      g.strokeRect(mcx0,mcy0,mcx1-mcx0,mcy1-mcy0);
      g.restore();
    });
  }

  // Box-zoom selection rectangle.
  if(mode==='box'&&boxStart&&boxCur){
    [g1,g2].forEach(function(g){
      g.save();
      g.strokeStyle='rgba(255,220,0,0.95)';g.lineWidth=1.5;g.setLineDash([5,3]);
      var bx=Math.min(boxStart.x,boxCur.x),by=Math.min(boxStart.y,boxCur.y);
      var bw=Math.abs(boxCur.x-boxStart.x),bh=Math.abs(boxCur.y-boxStart.y);
      g.strokeRect(bx,by,bw,bh);g.restore();
    });
  }

  // Axes on separate canvases: treatment Y left, projection Y right; X below both.
  drawYAxis(ya1,ch,zy0,zh,ih,true);
  drawYAxis(ya2,ch,zy0,zh,ih,false);
  drawXAxis(xa1,cw,zx0,zw,iw);
  drawXAxis(xa2,cw,zx0,zw,iw);
}

// Canvas coords → normalised image coords.
function c2n(cx,cy,cw,ch){
  return{x:zoom.x0+cx/cw*(zoom.x1-zoom.x0), y:zoom.y0+cy/ch*(zoom.y1-zoom.y0)};
}

function clampZoom(z){
  var minSz=0.02;
  var w=Math.max(minSz,z.x1-z.x0),h=Math.max(minSz,z.y1-z.y0);
  w=Math.min(1,w); h=Math.min(1,h);
  z.x0=Math.max(0,Math.min(z.x0,1-w)); z.y0=Math.max(0,Math.min(z.y0,1-h));
  z.x1=z.x0+w; z.y1=z.y0+h;
  return z;
}

function sendZoom(){
  clearTimeout(_zoomTimer);
  _zoomTimer=setTimeout(function(){
    setVal({offset_x:ox,offset_y:oy,zoom:{x0:zoom.x0,y0:zoom.y0,x1:zoom.x1,y1:zoom.y1}});
  },350);
}

function zoomBtn(dir){
  var step=args&&args.zoom_step?args.zoom_step:20;
  var factor=dir<0?(1-step/100):(1+step/100);
  var cx=(zoom.x0+zoom.x1)/2,cy=(zoom.y0+zoom.y1)/2;
  var w=(zoom.x1-zoom.x0)*factor,h=(zoom.y1-zoom.y0)*factor;
  zoom=clampZoom({x0:cx-w/2,y0:cy-h/2,x1:cx+w/2,y1:cy+h/2});
  draw();
  setVal({offset_x:ox,offset_y:oy,zoom:{x0:zoom.x0,y0:zoom.y0,x1:zoom.x1,y1:zoom.y1}});
}

function resetZoom(){
  zoom={x0:0,y0:0,x1:1,y1:1};
  draw();
  setVal({offset_x:ox,offset_y:oy,zoom:{x0:0,y0:0,x1:1,y1:1}});
}

function onDown(e,c){
  var r=c.getBoundingClientRect();
  var cx=e.clientX-r.left,cy=e.clientY-r.top;
  mx0=cx; my0=cy; drag=true; e.preventDefault();
  if(mode==='pan'){ox0=ox;oy0=oy;c.style.cursor='grabbing';}
  else if(mode==='box'){boxStart={x:cx,y:cy};boxCur={x:cx,y:cy};}
}
function onMove(e,c){
  if(!drag||!args)return;
  var r=c.getBoundingClientRect();
  var cx=e.clientX-r.left,cy=e.clientY-r.top;
  if(mode==='pan'){
    var zw=zoom.x1-zoom.x0,zh=zoom.y1-zoom.y0;
    ox=Math.round(ox0+(cx-mx0)/c.width*zw*args.img_w);
    // Negate cy delta: dragging up (cy decreases) increases oy → contour moves up.
    oy=Math.round(oy0-(cy-my0)/c.height*zh*args.img_h);
  } else if(mode==='box'){
    boxCur={x:cx,y:cy};
  }
  draw();
}
function onUp(e,c){
  if(!drag)return; drag=false;
  if(mode==='pan'){
    c.style.cursor='grab';
    if(ox!==ox0||oy!==oy0)_sendPos();
  } else if(mode==='box'){
    if(boxStart&&boxCur&&Math.abs(boxCur.x-boxStart.x)>8&&Math.abs(boxCur.y-boxStart.y)>8){
      var cw=c.width,ch=c.height;
      var n0=c2n(Math.min(boxStart.x,boxCur.x),Math.min(boxStart.y,boxCur.y),cw,ch);
      var n1=c2n(Math.max(boxStart.x,boxCur.x),Math.max(boxStart.y,boxCur.y),cw,ch);
      zoom=clampZoom({x0:n0.x,y0:n0.y,x1:n1.x,y1:n1.y});
      setVal({offset_x:ox,offset_y:oy,zoom:{x0:zoom.x0,y0:zoom.y0,x1:zoom.x1,y1:zoom.y1}});
    }
    boxStart=null; boxCur=null;
    draw();
  }
}

[c1,c2].forEach(function(c){
  c.addEventListener('mousedown',function(e){onDown(e,c);});
  c.addEventListener('mousemove',function(e){onMove(e,c);});
  c.addEventListener('mouseup',  function(e){onUp(e,c);});
  c.addEventListener('mouseleave',function(e){if(drag)onUp(e,c);});
  c.addEventListener('wheel',function(e){},{passive:true});
});

var _readyMsg={type:'streamlit:componentReady',apiVersion:1};
_post(_readyMsg);
var _readyTimer=setInterval(function(){
  if(_rendered){clearInterval(_readyTimer);return;}
  _post(_readyMsg);
},100);
